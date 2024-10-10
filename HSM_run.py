import os
import geopandas as gpd
import numpy as np
import pandas as pd
import flopy
import subprocess
import warnings
import time
import rascontrol
import tables as tb
from shapely.geometry import Polygon
import gc
import pythoncom
from joblib import Parallel, delayed
from shapely.geometry import  Point
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.neighbors import KNeighborsRegressor
import shutil
import concurrent.futures
from scipy.interpolate import interp1d
import win32com.client
import tables
import math

def hec_run(flow_write_dic, hec_route, version, Flow_file, run_time_day, swat_out_flow):
    subprocess.run(["taskkill", "/f", "/im", "Ras.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
    start_time = time.time()
    pythoncom.CoInitialize()
    
    def format_Q(Q):
        Q_converted = Q * 35.3147  # 유량 변환
        
        if abs(Q_converted) >= 100:
            # 절대값이 100 이상인 경우: 정수로 표시 (예: "113")
            formatted_Q = "{:.0f}".format(Q_converted)
        elif abs(Q_converted) >= 10:
            # 절대값이 10 이상 100 미만인 경우: 소수점 이하 한 자리 (예: "11.3")
            formatted_Q = "{:.1f}".format(Q_converted)
        elif abs(Q_converted) >= 1:
            # 절대값이 1 이상 10 미만인 경우: 소수점 이하 한 자리 (예: "3.2")
            formatted_Q = "{:.2f}".format(Q_converted)
        else:
            if Q_converted >= 0:
                # 절대값이 1 미만이고 양수인 경우: 소수점 이하 두 자리, 앞의 '0' 생략 (예: ".72")
                formatted_Q = "{:.3f}".format(Q_converted)[1:]
            else:
                # 절대값이 1 미만이고 음수인 경우: 소수점 이하 한 자리, 앞의 '0' 생략 (예: "-.7")
                formatted_Q = "-{:.2f}".format(abs(Q_converted))[1:]
        
        return formatted_Q

    path = hec_route +"."+Flow_file
    with open(path, "w") as ts:
        for item in flow_write_dic["first_datas"]:
            ts.write(item)
        for key, item_in in flow_write_dic.items():
            if key != "first_datas":
                try:
                    parts = list(map(int, key.split("-")))
                    Q=swat_out_flow.loc[parts[0]," Flow_out"]
                    for basin_idx in parts[1:]:
                        Q-= swat_out_flow.loc[basin_idx," Flow_out"]
                    
                    fo_Q = format_Q(Q)
                    data = item_in.format(Q=fo_Q)
                    ts.write(data)
                except:
                    if key !="end_datas" and key !="first_datas":
                        ts.write(item_in)
        for item in flow_write_dic["end_datas"]:
            ts.write(item)
            
    ras_controller = win32com.client.Dispatch(f"RAS641.HECRASController")

    ras_project_path = f"{hec_route}.prj"
    ras_controller.Project_Open(ras_project_path)
    ras_controller.Compute_CurrentPlan(None, None, True)
    
    ras_controller.Project_Close()
    ras_controller = None  # 객체 초기화

    subprocess.run(["taskkill", "/f", "/im", "Ras.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"     hec runtime: {elapsed_time:.2f} seconds")        

def river_interpolated(hec_route, Plan_file, Td_area_name, s_polygon_gdf, Dem_df, resolution, modflow_cell_grid):  
    def points_within_polygon(polygon, points_gdf):
        """공간 인덱스를 활용해 폴리곤 내에 포함된 포인트 찾기."""
        sindex = points_gdf.sindex
        possible_matches_index = list(sindex.intersection(polygon.bounds))
        possible_matches = points_gdf.iloc[possible_matches_index]
        return possible_matches[possible_matches.within(polygon)]
    
    if True:   # hdf 읽기
        path = f"{hec_route}.{Plan_file}.hdf"
        hdf_file = tables.open_file(path, mode='r')
        # HDF5 데이터 가져오기
        max_water_surface_path = f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{Td_area_name}/Maximum Water Surface'
        min_water_surface_path = f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{Td_area_name}/Minimum Water Surface'
        Surface_min_el = f'/Geometry/2D Flow Areas/Perimeter 1/Cells Minimum Elevation'
        coordinate_path = f'/Geometry/2D Flow Areas/{Td_area_name}/Cells Center Coordinate'
        
        max_water_surface_data = hdf_file.get_node(max_water_surface_path)[:]
        min_water_surface_data = hdf_file.get_node(min_water_surface_path)[:]
        coordinates_data = hdf_file.get_node(coordinate_path)[:]
        Surface_min_el = hdf_file.get_node(Surface_min_el)[:]
        hdf_file.close()
        
        # 수위 데이터 계산
        water_surface_data = (np.array(max_water_surface_data) + np.array(min_water_surface_data)) / 2
        hec_out_data = pd.DataFrame({
            'X': coordinates_data[:, 0],  
            'Y': coordinates_data[:, 1],  
            'EL': water_surface_data[0], 
            'DEPTH': water_surface_data[1],
            'Min_EL' : Surface_min_el
        })
        
        
        geometry = gpd.points_from_xy(hec_out_data["X"], hec_out_data["Y"])
        point_gdf = gpd.GeoDataFrame(hec_out_data, geometry=geometry)
        
        # 공간 결합 (벡터화 사용)
        joined_gdf = gpd.sjoin(s_polygon_gdf, point_gdf, how="inner", predicate="intersects")
        joined_gdf = joined_gdf[['index_right', "X", "Y", "Area", "geometry", "EL", 'DEPTH']].reset_index(drop=True)
        
        # 유효한 데이터 필터링
        filtered_gdf = joined_gdf[(joined_gdf['DEPTH'] > 0.005) & (joined_gdf.is_valid) & (~joined_gdf.is_empty)].reset_index(drop=True)
        
    if True: # 표고 위 부분 찾기
        from scipy.spatial import KDTree
        Dem_df['X'] = Dem_df.geometry.x
        Dem_df['Y'] = Dem_df.geometry.y
        
        # joined_gdf에서 X, Y, EL 값 추출
        points =filtered_gdf[['X', 'Y']].values  # 보간할 좌표들 (X, Y)
        values = filtered_gdf['EL'].values        # 보간할 값 (EL)
        
        # KDTree를 사용해 joined_gdf의 X, Y 좌표를 기반으로 최근접 이웃을 찾음
        tree = KDTree(points)
        
        # Dem_df의 각 포인트에 대해 최근접 이웃을 찾음
        dist, idx = tree.query(Dem_df[['X', 'Y']].values)
        
        # 최근접 이웃의 EL 값을 Dem_df의 새로운 컬럼에 할당
        threshold_distance = 50
        Dem_df['interpolated_EL'] = np.where(dist > threshold_distance, 0, values[idx])
        river_dem_df=Dem_df[Dem_df["interpolated_EL"]>Dem_df["value"]]
        river_dem_df = river_dem_df.reset_index()
        
        buffered = river_dem_df.buffer(resolution * np.sqrt(2) / 2)
        unified_polygon = buffered.unary_union

    if True: #river cell 찾고 값 입력
    
        simplify_tolerance = 1  # 단순화 허용 오차
        buffered_polygons = list(unified_polygon.geoms) if unified_polygon.geom_type == 'MultiPolygon' else [unified_polygon]
        simplified_buffered_polygons = [poly.simplify(simplify_tolerance) for poly in buffered_polygons]
        
        # 공간 인덱스 최적화
        buffered_sindex = gpd.GeoSeries(simplified_buffered_polygons).sindex
        modflow_intersections = modflow_cell_grid["geometry"].apply(lambda cell: list(buffered_sindex.intersection(cell.bounds)))
    
        river_input_cell = []
        
        # 모형 셀 교차 연산 최적화
        for idx, M_C_row in modflow_cell_grid.iterrows():
            cell = M_C_row["geometry"].simplify(simplify_tolerance)
            possible_matches_index = modflow_intersections[idx]
            possible_matches = [simplified_buffered_polygons[i] for i in possible_matches_index]
            total_intersection_area = sum(cell.intersection(poly).area for poly in possible_matches)
            if total_intersection_area > 0:
                M_C_row["buffered_area"] = total_intersection_area
                river_input_cell.append(M_C_row)
        
        modflow_input_gdf = gpd.GeoDataFrame(river_input_cell)
        
        # 포인트와 교차된 셀들에 대한 평균 EL 계산 최적화
        river_sindex = river_dem_df.sindex
        modflow_intersections_points = modflow_input_gdf["geometry"].apply(lambda cell: list(river_sindex.intersection(cell.bounds)))
        
        mod_river_input = []
        for idx, row_mod in modflow_input_gdf.iterrows():
            if len(modflow_intersections_points[idx]) > 0:
                row = modflow_intersections_points[idx]
                sum_EL = sum(river_dem_df.loc[poly, "interpolated_EL"] for poly in row)
                average = sum_EL / len(row) if len(row) > 0 else 0
                row_mod["water_stage"] = average
                sum_EL = sum(river_dem_df.loc[poly, "value"] for poly in row)
                average = sum_EL / len(row) if len(row) > 0 else 0
                row_mod["bottom_el"] = average
                mod_river_input.append(row_mod)
        
        gdf = gpd.GeoDataFrame(mod_river_input)
        mod_river_input = pd.DataFrame(mod_river_input)
        
    return mod_river_input

def mod_run(wel_time_dic, mod_river_input, run_time_day, wel, bas_dic, mf, river_conductivity, river_thickness, TXTINOUT, modflow_swat_hru, rch, output_data_space, cols_flat, rows_flat, model_layer):
    start_time = time.time()
    def process_well_data(run_time_day, wel_time_dic, wel):
        new_wel_data = []
        for idx, times in enumerate(wel_time_dic.keys()):
            if idx == len(wel_time_dic)-1:
                time_key = times
            elif run_time_day < times[1] and run_time_day >times[0]:
                time_key = times
                break
        for cell in range(len(wel_time_dic[time_key])): 
            layer, row, col = wel_time_dic[time_key]["k"][cell], wel_time_dic[time_key]["i"][cell], wel_time_dic[time_key]["j"][cell]
            new_q = wel_time_dic[time_key]["flux"][cell]
            new_wel_data.append((layer, row, col, new_q))
        wel.stress_period_data[0] = new_wel_data
        
        return wel
    
    def process_river_data(mod_river_input, river_conductivity, river_thickness, mf):
        layer = 0
        rows = np.array(mod_river_input["row"].tolist()) - 1
        cols = np.array(mod_river_input["col"].tolist()) - 1
        stage = mod_river_input["water_stage"].tolist()
        rbot = mod_river_input["bottom_el"].tolist()
        cond = np.array(mod_river_input["buffered_area"].tolist()) * river_conductivity / river_thickness
        
        riv_data = []
        for row, col, stg, rbt, cnd in zip(rows, cols, stage, rbot, cond):
            riv_data.append([layer, row, col, stg, cnd, rbt])
        riv_spd = {0: riv_data}
        
        riv = flopy.modflow.ModflowRiv(mf, ipakcb=53, stress_period_data=riv_spd)
        return riv
    
    def process_recharge_data(modflow_swat_hru, TXTINOUT, rch):
        Swat_recharge_data_route = os.path.join(TXTINOUT, "recharge_evap.modf")
        recharge_evap_data = pd.read_csv(Swat_recharge_data_route)
        recharge_evap_data.set_index(['HRU'], inplace=True)
        
        for grid, hru in modflow_swat_hru.items():
            recharge = 0
            for i in hru.index:
                recharge += recharge_evap_data.loc[hru.loc[i, "HRU_ID"], "Recharge"] * hru.loc[i, "overlap_area"]
            recharge = recharge / hru.loc[i, "grid_area"]
            rch[hru.loc[i, "row"]-1, hru.loc[i, "col"]-1] = recharge / 1000
        
        rch_spd = {0: rch}
        rch_mod = flopy.modflow.ModflowRch(mf, ipakcb=54, rech=rch_spd)
        return rch_mod
    
    # 병렬 실행
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(process_well_data, run_time_day, wel_time_dic, wel))
        futures.append(executor.submit(process_river_data, mod_river_input, river_conductivity, river_thickness, mf))
        futures.append(executor.submit(process_recharge_data, modflow_swat_hru, TXTINOUT, rch))
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        wel, riv, rch_mod = results

    if run_time_day!=1:
        head_file = flopy.utils.HeadFile(f"{mf.model_ws}/{mf.name}.hds")
        head_data = head_file.get_data(kstpkper=head_file.get_kstpkper()[-1])
        bas = flopy.modflow.ModflowBas(mf, ibound=bas_dic["Bnd"], strt=head_data)
    
    bas = flopy.modflow.ModflowBas.load(f"{mf.model_ws}/{mf.name}.bas",mf)
    mf.write_input()
    success, mfoutput = mf.run_model(silent=True)
    if not success:
        print("MODFLOW did not terminate normally.")
        
    head_file = flopy.utils.HeadFile(f"{mf.model_ws}/{mf.name}.hds")
    
    cbc = flopy.utils.CellBudgetFile(f"{mf.model_ws}/{mf.name}.RIV.cbc")    
    river_leakage = cbc.get_data(text="RIVER LEAKAGE")
    leakage_df = pd.DataFrame({
        'layer': np.repeat(np.arange(river_leakage[0].shape[0]), river_leakage[0].shape[1] * river_leakage[0].shape[2])+1,
        'row': np.tile(np.repeat(np.arange(river_leakage[0].shape[1]), river_leakage[0].shape[2]), river_leakage[0].shape[0])+1,
        'col': np.tile(np.arange(river_leakage[0].shape[2]), river_leakage[0].shape[0] * river_leakage[0].shape[1])+1,
        'leakage': river_leakage[0].flatten()})
    
    leakage_df = leakage_df[leakage_df['leakage'] != 0]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"    mod runtime: {elapsed_time:.2f} seconds")
    
    return head_file, rch, leakage_df
    
def ground_water_out(intersect_modflow_cell, leakage_df, drain_out_weight, weir_ground_out, ground_Q, swat_out_flow):
    for key, values in intersect_modflow_cell.items():
        merged_df = pd.merge(values, leakage_df, on=['row', 'col'], how='left')
        merged_df.fillna(0, inplace=True)
        merged_df['weighted_leakage'] = merged_df['leakage'] * merged_df['area']
        total_weighted_leakage = merged_df['weighted_leakage'].sum()
        total_area_sum = sum(list(merged_df.loc[:,"area"]))
        ground_Q[key] = total_weighted_leakage
        
        if total_area_sum!=0:
            weir_ground_out[key]=total_weighted_leakage/total_area_sum * drain_out_weight
        
    for key, item in ground_Q.items():
        swat_out_flow.loc[key," Flow_out"]=swat_out_flow.loc[key," Flow_out"]-item/86400
        
    return swat_out_flow, weir_ground_out

def zone_budget_v1(mf):
    cbc = flopy.utils.CellBudgetFile(f"{mf.model_ws}/{mf.name}.cbc")
    record_names = cbc.get_unique_record_names()

    cbc_pd = pd.DataFrame()
    for record in record_names:
        record_name = record.strip()
        name = str(record_name).strip("b'")
        data = cbc.get_data(text=name)
        data = data[0]
        total_inflow = data[data > 0].sum()
        total_outflow = data[data < 0].sum()
        new_row = pd.DataFrame([[name, total_inflow, total_outflow]], columns=['m3/day', 'IN', 'OUT'])
        cbc_pd = pd.concat([cbc_pd, new_row], ignore_index=True)

    # RIVER LEAKAGE 처리
    if "RIVER LEAKAGE" not in list(cbc_pd.index):
        cbc_riv = flopy.utils.CellBudgetFile(f"{mf.model_ws}/{mf.name}.RIV.cbc")
        data = cbc_riv.get_data(text="RIVER LEAKAGE")[0]
        total_inflow = data[data > 0].sum()
        total_outflow = data[data < 0].sum()
        new_row = pd.DataFrame([["River LEAKAGE", total_inflow, total_outflow]], columns=['m3/day', 'IN', 'OUT'])
        cbc_pd = pd.concat([cbc_pd, new_row], ignore_index=True)
        
    if "RECHARGE" not in list(cbc_pd.index):
        cbc_rch = flopy.utils.CellBudgetFile(f"{mf.model_ws}/{mf.name}.RCH.cbc")
        data = cbc_rch.get_data(text="RECHARGE")[0]
        total_inflow = data[data > 0].sum()
        total_outflow = data[data < 0].sum()
        new_row = pd.DataFrame([["RECHARGE", total_inflow, total_outflow]], columns=['m3/day', 'IN', 'OUT'])
        cbc_pd = pd.concat([cbc_pd, new_row], ignore_index=True)
    
    return cbc_pd

def zone_budget_v2(mf):
    path =mf.model_ws
    name =mf.name
    
    list_path = path+"/"+name+".list"
    
    zone_start_line = 0
    with open(list_path, "r") as file:
        lines = file.readlines()
        for idx in range(len(lines)-1,-1,-1):
            if "     CUMULATIVE VOLUMES      L**3       RATES FOR THIS TIME STEP      L**3/T" in lines[idx]:
                zone_start_line = idx
                break
    
    cbc_pd=pd.DataFrame()
    for idx in range(zone_start_line+3,100000):
        if " PERCENT DISCREPANCY" in lines[idx]:
            break
        
        if "IN:" in lines[idx]:
            col_name = "IN"
        elif "OUT:" in lines[idx]:
            col_name = "OUT"
        elif "=" in lines[idx]:
            parts = lines[idx].split("=")
            z_name = parts[0].replace(" ","")
            if "TOTAL" in lines[idx]:
                z_name = "TOTAL"
            cbc_pd.loc[z_name, col_name] = float(parts[-1])
            
    return cbc_pd
    
def convergence(TXTINOUT, run_time_day, flow_write_dic, hec_route, version, Flow_file, Plan_file, Td_area_name, s_polygon_gdf, Dem_df, resolution, modflow_cell_grid, wel_time_dic, wel, bas_dic, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, output_data_space, cols_flat, rows_flat, model_layer, leakage_df):
    ground_Q = {}
    weir_ground_out={}
    
    swat_out_flow=pd.read_csv(TXTINOUT+"/basin_flow.modf")
    swat_out_flow.set_index('Basin', inplace=True)
    
    hec_run(flow_write_dic, hec_route, version, Flow_file, run_time_day, swat_out_flow)  
    mod_river_input = river_interpolated(hec_route, Plan_file, Td_area_name, s_polygon_gdf, Dem_df, resolution, modflow_cell_grid)
    head_file, rch, leakage_df = mod_run(wel_time_dic, mod_river_input, run_time_day, wel, bas_dic, mf, river_conductivity, river_thickness, TXTINOUT, modflow_swat_hru, rch, output_data_space, cols_flat, rows_flat, model_layer)

    #-------------------------------------------------------------------------------------------hec 결과 저장
    def save_hec_data(run_time_day, final_df, output_data_space):
        out_hec_data = os.path.join(output_data_space, "hec_data")
        if not os.path.exists(out_hec_data):
            os.makedirs(out_hec_data)
    
        txt_file_name = os.path.join(out_hec_data, f"day_{run_time_day}.csv")
        selected_columns = ['River', 'Reach', "RX", "min_elevation", "Water Surface", "Flow_Q"]
        final_df_selected = final_df[selected_columns]
        final_df_selected.to_csv(txt_file_name, index=False)
    
    def save_head_data(run_time_day, rows_flat, cols_flat, head_file, model_layer, output_data_space):
        head_out_data_path = os.path.join(output_data_space, "head")
        if not os.path.exists(head_out_data_path):
            os.makedirs(head_out_data_path)
    
        head_df = pd.DataFrame({'row': rows_flat, 'col': cols_flat})
        head_data_list = []
    
        for nlay in range(model_layer):
            head_data = head_file.get_data(kstpkper=(0, 0), mflay=nlay)
            head_data_flat = head_data.flatten()
            head_data_list.append(head_data_flat)
    
        for nlay in range(model_layer):
            head_df[f'head_{nlay + 1}_{run_time_day}'] = head_data_list[nlay]
    
        output_csv_path = os.path.join(head_out_data_path, f"{run_time_day}.csv")
        head_df.to_csv(output_csv_path, index=False, header=True)
    
    def save_recharge_data(run_time_day, rch, output_data_space):
        recharge_out_data_path = os.path.join(output_data_space, "recharge")
        if not os.path.exists(recharge_out_data_path):
            os.makedirs(recharge_out_data_path)
    
        reshaped_rch = pd.DataFrame([(i + 1, j + 1, rch[i, j]) for i in range(rch.shape[0]) for j in range(rch.shape[1])],
                                    columns=['row', 'col', 'value'])
        reshaped_rch['value'] = reshaped_rch['value'] * 1000
        reshaped_rch = reshaped_rch.rename(columns={"value": "recharge_" + str(run_time_day)})
    
        output_csv = os.path.join(recharge_out_data_path, str(run_time_day) + ".csv")
        reshaped_rch.to_csv(output_csv, index=False, header=True)
    
    def save_river_stage(run_time_day, river_cell_data, output_data_space):
        river_out_data_path = os.path.join(output_data_space, "river")
        if not os.path.exists(river_out_data_path):
            os.makedirs(river_out_data_path)
    
        write_river = river_cell_data[["row", "col", "interpolated_mean"]].rename(columns={"interpolated_mean": "river_head_" + str(run_time_day)})
        output_csv = os.path.join(river_out_data_path, str(run_time_day) + ".csv")
        write_river.to_csv(output_csv, index=False, header=True)
    
    def copy_model_output_files(run_time_day, mf, output_data_space):
        MD_out_data_path = os.path.join(output_data_space, "MF_out")
        if not os.path.exists(MD_out_data_path):
            os.makedirs(MD_out_data_path)
    
        cbc_source_file = f"{mf.model_ws}/{mf.name}.cbc"
        hds_source_file = f"{mf.model_ws}/{mf.name}.hds"
        cbc_river_source_file = f"{mf.model_ws}/{mf.name}.RIV.cbc"
    
        shutil.copy(cbc_source_file, os.path.join(MD_out_data_path, f"{run_time_day}.cbc"))
        shutil.copy(hds_source_file, os.path.join(MD_out_data_path, f"{run_time_day}.hds"))
        shutil.copy(cbc_river_source_file, os.path.join(MD_out_data_path, f"{run_time_day}.RIV.cbc"))
    
    def save_cbc_data(run_time_day, mf, output_data_space):
        cbc_out_data_path = os.path.join(output_data_space, "cbc")
        if not os.path.exists(cbc_out_data_path):
            os.makedirs(cbc_out_data_path)
        
        cbc_pd = zone_budget_v2(mf)

        output_csv = os.path.join(cbc_out_data_path, f"{run_time_day}.csv")
        cbc_pd.to_csv(output_csv, index=True, header=True)
    
    def swat_out_data(TXTINOUT,output_data_space,run_time_day):
        basin_flow = os.path.join(TXTINOUT, "basin_flow.modf")
        recharge_evap = os.path.join(TXTINOUT, "recharge_evap.modf")
        
        path = os.path.join(output_data_space, "swat_out")
        if not os.path.exists(path):
            os.makedirs(path)
            
        basin_flow_copy = os.path.join(path, f"BF_{run_time_day}.txt")
        recharge_evap_copy = os.path.join(path, f"RE_{run_time_day}.txt")
        
        shutil.copy(basin_flow, basin_flow_copy)
        shutil.copy(recharge_evap, recharge_evap_copy)
        
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(save_hec_data, run_time_day, final_df, output_data_space))
        futures.append(executor.submit(save_head_data, run_time_day, rows_flat, cols_flat, head_file, model_layer, output_data_space))
        futures.append(executor.submit(save_recharge_data, run_time_day, rch, output_data_space))
        futures.append(executor.submit(save_river_stage, run_time_day, river_cell_data, output_data_space))
        futures.append(executor.submit(copy_model_output_files, run_time_day, mf, output_data_space))
        futures.append(executor.submit(save_cbc_data, run_time_day, mf, output_data_space))  # 물수지 저장 추가
        futures.append(executor.submit(swat_out_data, TXTINOUT,output_data_space,run_time_day))

        concurrent.futures.wait(futures)
    
    return leakage_df
        
def run_HSM(TXTINOUT, write_result, hec_route, version, Flow_file, sub_program_route, Plan_file, river_gdfs, river_horizontal_distance, Dem_df, resolution, modflow_cell_grid, \
            swat_hec_link_dic, output_data_space, final_interpolated_gdf, wel_time_dic, bas_dic, wel, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, cols_flat, rows_flat, model_layer, weir_dic, intersect_modflow_cell, drain_out_weight, nearest_river_weir, basin_to_river):
    run_time_day=0
    f_final_df=0
    leakage_df=0
    sstart_time = time.time()
    while True:
        process = subprocess.Popen([os.path.join(TXTINOUT, "swat_ver2.exe")],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)    
        os.chdir(TXTINOUT)
        while True:
            one_day_time = time.time()
            output = process.stdout.readline()
            if output:
                print(output.strip())
                # 엔터를 입력할 시 프로그램에 전달
                if "swat runday" in output:
                    run_time_day+=1
                    f_final_df, final_interpolated_gdf, leakage_df \
                    =convergence(run_time_day, TXTINOUT, write_result, hec_route, version, Flow_file, sub_program_route, Plan_file, river_gdfs, river_horizontal_distance, Dem_df, resolution, modflow_cell_grid, swat_hec_link_dic, \
                     output_data_space, final_interpolated_gdf, wel_time_dic, bas_dic, wel, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, cols_flat, rows_flat, model_layer, weir_dic, intersect_modflow_cell,\
                     drain_out_weight, nearest_river_weir, f_final_df, basin_to_river, leakage_df)
                        
                    process.stdin.write("\n")
                    process.stdin.flush()
                    one_day_time_stop = time.time()
                    print(f"  Run_time : {one_day_time_stop - one_day_time}")
            # 프로세스가 종료되었는지 확인
            if process.poll() is not None:
                break
        subprocess.run(["taskkill", "/f", "/im", "swat_ver2.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if run_time_day > 1:
            eend_time = time.time()
            elapsed_time = eend_time - sstart_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Total runtime: {int(hours)} hours, {int(minutes)} minutes,{seconds:.2f} seconds")
            break