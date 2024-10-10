import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import sys
import flopy
import math
import openpyxl
import warnings
import gc
import re
import shutil
import swatpy
import glob
import pickle
import tables
from raspy_auto import Ras, API
from shapely.geometry import Point, LineString
from rasterio.mask import mask
from rasterio.transform import xy
from shapely.geometry import Point, Polygon

def Hec_Data_Read(HecData):
    with open(HecData + ".prj", "r") as hec:
        lines = hec.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if "Current Plan" in line:
                a = line.split("=")
                Plan_file = a[-1]
                break
    Plan_path = HecData + "." + str(Plan_file)

    # Reading the plan file
    with open(Plan_path,"r") as plan:
        liness = plan.readlines()
        for idx, line in enumerate(liness):
            line = line.strip()  # 줄 끝의 엔터 문자를 제거
            if "Program Version" in line:
                a=line.split("=")
                version = a[-1]
                version = version.replace('.', '')
                formatted_version = "{}.{}.{}".format(version[0], version[1], version[2])
            if "Geom File" in line:
                a=line.split("=")
                Geom_file = a[-1]
            if "Simulation Date" in line:
                a=line.split("00,")
                end_day = a[-1].replace("\n", "")
                end_day = end_day.replace(",", " ")+":00"
            if "Flow File" in line:
                a=line.split("=")
                Flow_file = a[-1]
                break
    del liness, lines
    
    return Geom_file, Flow_file, version, formatted_version, Plan_file, end_day

def shape_file_read(Python_need_shp_folder, HecData, Plan_file, Td_area_name):
    from scipy.spatial import Delaunay
    shp_route={}
    for i in ["HRU","Watershed", "Hec_cells"]:
        shp_route[i] = gpd.read_file(os.path.join(Python_need_shp_folder,i)+".shp")
    
    origin_shape_gdf = shp_route["Hec_cells"]
    s_polygon_gdf = origin_shape_gdf.explode(index_parts=True)
    
    return shp_route , origin_shape_gdf, s_polygon_gdf

def raster_dem_read(Python_need_shp_folder, origin_shape_gdf):
    files_in_folder = os.listdir(Python_need_shp_folder)
    tiff_files = [file for file in files_in_folder if file.endswith('.tif') or file.endswith('.tiff')]
    raster_path = os.path.join(Python_need_shp_folder, tiff_files[0])
    
    # 3. 레스터 파일을 포인트로 변환
    with rasterio.open(raster_path) as src:
        resolution = src.res[0]
        crs = src.crs
        # 3. 폴리곤을 GeoJSON 형식으로 변환 (rasterio가 GeoJSON 형식을 사용)
        polygons = [origin_shape_gdf.geometry.values[0]]  # 첫 번째 폴리곤을 사용 (여러 개일 경우 리스트로 처리 가능)
        
        # 4. 레스터 클립 (mask 함수 사용하여 폴리곤 외부를 NaN으로 처리)
        out_image, out_transform = mask(src, polygons, crop=True, nodata=np.nan)  # NaN으로 마스킹 처리
        
        # 5. 클립된 레스터의 메타데이터 업데이트
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": np.nan  # NaN 값 설정
        })
        
        # 5. 클립된 레스터를 포인트화 시키기
        rows, cols = out_image.shape[1], out_image.shape[2]
        
        points_data = []
        
        # 각 포인트(픽셀)를 반복 처리
        for row in range(rows):
            for col in range(cols):
                value = out_image[0, row, col]
                
                if not np.isnan(value):  # NaN이 아닌 값만 처리
                    # 좌표 변환
                    x, y = xy(out_transform, row, col)
                    points_data.append({'geometry': Point(x, y), 'value': value})
        
        # 포인트를 GeoDataFrame으로 변환
        raster_point_gdf = gpd.GeoDataFrame(points_data)
        Dem_df = raster_point_gdf.dropna(subset=['value'])
        
    return Dem_df, crs, resolution

def Modflow_read(modflow_model_name, ModData, exe_path, initial_head, Modflow_work_space):
    warnings.filterwarnings("ignore", category=UserWarning)
    mf = flopy.modflow.Modflow(modflow_model_name, exe_name=exe_path, version='mfnwt', model_ws=ModData)
    all_data = [
        "Bas",  # Basic Package (BAS)
        "Dis",  # Discretization Package (DIS)`
        "Lpf",  # Layer Property Flow Package (LPF)
        "Bcf",  # Block-Centered Flow Package (BCF)
        "Rch",  # Recharge (RCH)
        "Evt",  # Evapotranspiration (EVT)
        "Ghb",  # General-Head Boundary (GHB)
        "Riv",  # River (RIV)
        "Drn",  # Drain (DRN)
        "Sfr",  # Streamflow-Routing (SFR)
        "Mnw2", # Multi-Node Well (MNW2)
        "Drt",  # Drain Return Flow (DRT)
        "Nwt",  # NWT Solver Package (NWT)
        "Upw",  # UPW Package
        "Uzf",  # UZF Package
        "Rip",  # Riparian Evapotranspiration (RIP)
        "Sub",  # Subsidence Package (SUB)
        "VMG",  # 격자 좌표
        "VMO",
        "Wel",
        "Hfb"]  # 관측정 위치
    
    data = []
    for filename in os.listdir(ModData):
        for mod in all_data:
            if "."+mod.lower() in filename.lower():     
                data.append(mod)
            
    data_routes={}
    except_file=[]
    ModflowClass={}
    for i in data:
        try:
            data_routes[i] = os.path.join(ModData, f'{modflow_model_name}.{i.upper()}')
            class_name = "Modflow" + i
            ModflowClass[i] = getattr(__import__("flopy.modflow", fromlist=[class_name]), class_name)
            globals()[i.lower()] = ModflowClass[i].load(data_routes[i], model=mf)
        except:
            if i not in ["Riv","Dis","VMG","VMO","Bas","Upw","Wel"]:
                except_file.append(i)
            pass
    
    oc_before = mf.get_package("OC")
    dis = mf.get_package('DIS')
    model_row = dis.nrow
    total_col=np.sum(dis.delr.array)
    col_size=dis.delc[0]
    model_col=dis.ncol
    total_row=np.sum(dis.delc.array)
    model_layer = dis.nlay
    
    
    nlay = dis.nlay
    nrow = dis.nrow
    ncol = dis.ncol
    top = dis.top
    botm = dis.botm
    delr = dis.delr
    delc = dis.delc
    
    # 스트레스 기간 관련 설정만 변경하면서 새로운 DIS 객체 생성
    new_dis = flopy.modflow.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=1,  # 스트레스 기간 수를 1로 설정
        perlen=[1.0],  # 스트레스 기간 길이
        nstp=[6],  # 시간 단계 수
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        steady= [False])
    
    period_lengths = dis.perlen.array  # 각 스트레스 기간의 길이 (일 단위)
    #------------------------------------------------- bas 와 upw 제작------------------------
    wel = mf.get_package('Wel')
    wel_data_origin = wel.stress_period_data.data
    wel_time_dic={}
    times=0
    b_time = 0
    for idx,key in enumerate(period_lengths):
        times+=float(period_lengths[idx])
        wel_time_dic[(b_time,times)]=wel_data_origin[idx]  # [0]["j"] 이런식
        b_time = times
    
    with open(data_routes["VMG"], "r") as vmg: # 격자 좌표정보
        V_count=0
        for num, line in enumerate(vmg):
            parts = line.split()
            if num==0:
                V_count+=float(parts[0])+3
            elif num==1:
                xll=float(parts[0])
                xul=xll+total_col
            elif num==V_count:
                yll=float(parts[0])
                yul=yll+total_row
                break

    head_file = flopy.utils.HeadFile(initial_head)
    head_data = head_file.get_data(kstpkper=head_file.get_kstpkper()[-1])
    
    with open(data_routes["Bas"], "r") as bas: # inactive cell, 초기 조건
        write=False
        bas_data=["Bnd","Head"]
        bas_dic={}
        for i in bas_data:
            bas_dic[i]=np.zeros((model_layer,model_row,model_col))
        for num, line in enumerate(bas):
            parts = line.split()
            parts = [part.replace(' ','') for part in parts]
            if "Layer" in parts:
                write=parts[-4]
                layer_num=int(parts[-1])-1
                count=0
                row=0
            elif write!=False:
                for data in parts:
                    bas_dic[write][layer_num][row][count]=float(data)
                    count+=1
                    if count==model_col:
                        count=0
                        row+=1
                        if row==model_row:
                            write=False
        bas = flopy.modflow.ModflowBas(mf, ibound=bas_dic["Bnd"], strt=head_data)
        bas_dic["Bnd"] = bas_dic["Bnd"].astype(np.int8)    
    
    with open(data_routes["Upw"], "r") as upw:  # 수리 상수
        write=False
        hy_data=["Specific storage","Specific yield","Conductivity along", "VKA: Kz"]
        hy_dic={}
        for i in hy_data:
            hy_dic[i]=np.zeros((model_layer,model_row,model_col))
        for line in upw:
            parts = line.split()
            parts = [part.replace(' ','') for part in parts]
            if "layer" in parts:
                write=parts[4]+" "+parts[5]
                layer_num=int(parts[-1])-1
                count=0
                row=0
                if int(parts[0]) == 0:
                    hy_dic[write][layer_num, :, :]=parts[1]
            elif write!=False:
                for data in parts:
                    hy_dic[write][layer_num][row][count]=float(data)
                    count+=1
                    if count==model_col:
                        count=0
                        row+=1
        upw = flopy.modflow.ModflowUpw(mf,  hk=hy_dic["Conductivity along"],  vka=hy_dic["VKA: Kz"],
            ss=hy_dic["Specific storage"], sy=hy_dic["Specific yield"], laytyp=1, hdry=999, iphdry=1)
        
        hy_dic["hk"] = hy_dic["Conductivity along"]
        hy_dic["vka"] = hy_dic["VKA: Kz"]
        hy_dic["ss"] = hy_dic["Specific storage"]
        hy_dic["sy"] = hy_dic["Specific yield"]
        
        del hy_dic["Conductivity along"]
        del hy_dic["VKA: Kz"]
        del hy_dic["Specific storage"]
        del hy_dic["Specific yield"]
        
    if "Rch" in except_file:   # 함양량
        write=False
        recharge = {}
        with open(data_routes["Rch"], "r") as rch:
            period=-1
            for line in rch:
                parts = line.split()
                parts = [part.replace(' ','') for part in parts]
                if "Recharge" in parts:
                    write=True
                    count=0
                    row=0
                    period+=1
                    rch_np = np.zeros((model_row,model_col))
                    recharge[period] = rch_np
                elif write!=False:
                    for data in parts:
                        print()
                        recharge[period][row][count]=float(data)
                        count+=1
                        if count==model_col:
                            count=0
                            row+=1
                            if row==model_row:
                                write=False
        rch = flopy.modflow.ModflowRch(mf, rech=recharge)
    
        # 스트레스 기간 데이터 설정
    stress_period_data = {(0, 0): ['SAVE BUDGET', 'SAVE HEAD', 'SAVE DRAWDOWN', 'PRINT BUDGET'], (0, 5): ['SAVE BUDGET', 'SAVE HEAD', 'SAVE DRAWDOWN', 'PRINT BUDGET']}

    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data=stress_period_data,
        compact=False,
        unitnumber=[14, 150, 151, 52, 53])

    
    rch= mf.get_package('RCH')
    mf.change_model_ws(Modflow_work_space)
    mf.write_input()
    
    with open(data_routes["VMO"], "r") as VMO:  #관측정
        obs_data=[]
        well_num=0
        for num, line in enumerate(VMO):
            if num==1:
                parts=line.split("=")
                obs_num=parts[-1]
            elif "Well=" in line:
                parts=line.split("=")
                well_name = parts[-1].replace("'", "")
                well_name = well_name.replace("\n", "")
                well_num+=1
            elif "X=" in line:
                parts=line.split(" ")
                parts = [part.replace(' ','') for part in parts]
                obs_col=math.ceil(abs(float(parts[-1])-xll)/col_size)
            elif "Y=" in line:   
                parts=line.split(" ")
                parts = [part.replace(' ','') for part in parts]
                obs_row=math.ceil(abs(float(parts[-1])-yul)/col_size)
            elif "Min" not in line and "Max" not in line and "Z=" in line:
                parts=line.split(" ")
                parts = [part.replace(' ','') for part in parts]
                obs_z=float(parts[-1])
                for layer in range(model_layer):
                    if obs_z >= dis.botm.array[layer,obs_row-1,obs_col-1]:
                        obs_layer=layer+1
            elif "end Well" in line:
                txt = f"{well_name} {obs_row} {obs_col} {obs_layer}\n"
                obs_data.append(txt)
    
    output_file = os.path.join(Modflow_work_space, "obs_data.txt")
    with open(output_file, "w") as f:
        f.writelines(obs_data)  # 리스트 내용을 파일에 쓰기
    
    return mf, wel, wel_time_dic, bas_dic, model_col,xll, xul, yll, yul, dis, bas, rch, model_col, model_row, model_layer, hy_dic
    
def Hec_river_write(Flow_file, HecData):
    path = HecData +"."+Flow_file
    write_data_list = []
    flow_write_dic = {"first_datas": [],"end_datas": []}
    with open(path, "r") as ts:
        start_data = False
        end_start = False
        lines = ts.readlines()
        for idx, line in enumerate(lines):
            if "Boundary Location" in line:
                start_data = True
                end_start = False
                parts = line.split(",")
                key = parts[-2].replace('\n', '')
                if "out" in line:
                    new_line = "Friction Slope=0.01,0\n"
                    write_data_list.append(line + new_line)
                    flow_write_dic[key] = line + new_line
                else:
                    new_line = (
                        "Interval=1WEEK\n"
                        "Flow Hydrograph= 2\n"
                        "    {Q}    {Q}\n"
                        "Stage Hydrograph TW Check=1\n"
                        "Flow Hydrograph Slope= 0\n"
                        "DSS Path=\n"
                        "Use DSS=False\n"
                        "Use Fixed Start Time=False\n"
                        "Fixed Start Date/Time=,\n"
                        "Is Critical Boundary=False\n"
                        "Critical Boundary Flow=\n"
                    )
                    write_data_list.append(line + new_line)
                    flow_write_dic[key] = line + new_line
                    
            elif "Critical Boundary Flow=" in line:
                end_start = True
                    
            elif end_start == True:
                flow_write_dic["end_datas"].append(line)
                    
            elif start_data == False:
                write_data_list.append(line)
                flow_write_dic["first_datas"].append(line)
            

    
    return flow_write_dic

def modflow_swat_linker(shp_route, xll, yll, xul, yul, bas_dic, dis, model_col, sub_program_route, crs):
    modflow_cell_grid_path = os.path.join(sub_program_route, "modflow_cell_grid.shp")

    def create_fishnet(bounds, dis):
        minx, miny, maxx, maxy = bounds
        polygons = []
        rows = len(dis.delc)
        cols = len(dis.delr)

        for i in range(rows):
            for j in range(cols):
                x1 = minx + np.sum(dis.delr.array[:j])
                y1 = maxy - np.sum(dis.delc.array[:i])
                x2 = minx + np.sum(dis.delr.array[:j+1])
                y2 = maxy - np.sum(dis.delc.array[:i+1])
                polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        return gpd.GeoDataFrame({'geometry': polygons}, crs= crs), rows, cols

    # 경계 값 설정 및 Fishnet 생성
    bounds = [xll, yll, xul, yul]
    modflow_cell_grid, rows, cols = create_fishnet(bounds, dis)

    # 셀 속성 추가
    modflow_cell_grid['grid_id'] = modflow_cell_grid.index
    modflow_cell_grid["grid_area"] = modflow_cell_grid.geometry.area
    modflow_cell_grid["row"] = modflow_cell_grid['grid_id'] // model_col + 1
    modflow_cell_grid["col"] = modflow_cell_grid['grid_id'] % model_col + 1
    
    # 비활성 셀 제거
    active_cells_mask = bas_dic["Bnd"][0][modflow_cell_grid['row'] - 1, modflow_cell_grid['col'] - 1] != 0
    modflow_cell_grid = modflow_cell_grid[active_cells_mask]
    modflow_cell_grid.to_file(modflow_cell_grid_path, driver='ESRI Shapefile')
    modflow_cell_grid = modflow_cell_grid.reset_index(drop=True)
    
    # MODFLOW grid와 HRU의 교차 영역 계산 (병렬 처리 제거)
    dhru_grid = gpd.overlay(modflow_cell_grid, shp_route["HRU"], how='intersection')
    
    dhru_grid["overlap_area"] = dhru_grid.geometry.area
    dhru_grid_data = dhru_grid[['grid_id', 'grid_area', 'row', 'col', 'overlap_area', "SUBBASIN", "HRUGIS", "GRIDCODE"]]
    modflow_swat_hru = dhru_grid_data.sort_values(by='grid_id')
   
    return modflow_cell_grid, modflow_swat_hru

def modcell_basin(modflow_cell_grid, shp_route):
    basin_shape = shp_route["Watershed"]
    
    for idx, shape in basin_shape.iterrows():
        if not shape.geometry.is_valid:
            basin_shape.loc[idx, "geometry"] = shape.geometry.buffer(0)
    
    intersect_modflow_cell = []

    # 각 MODFLOW 셀에 대해 교차 여부와 면적 계산
    for i, modflow_row in modflow_cell_grid.iterrows():
        # 교차하는 shp_route의 폴리곤과 그 면적을 계산
        intersecting_shapes = basin_shape[basin_shape.intersects(modflow_row.geometry)]
        
        for j, shp_row in intersecting_shapes.iterrows():
            # 교차하는 면적 계산
            intersection = modflow_row.geometry.intersection(shp_row.geometry)
            if not intersection.is_empty:
                intersection_area = intersection.area
                result = {
                    'row': modflow_row["row"],
                    'col': modflow_row["col"],
                    'area': intersection_area / modflow_row["grid_area"],
                    'basin_name': shp_row["Subbasin"],
                }
                intersect_modflow_cell.append(result)
    
    intersect_modflow_cell = pd.DataFrame(intersect_modflow_cell)
    intersect_modflow_cell = {basin_name: group for basin_name, group in intersect_modflow_cell.groupby('basin_name')}
    return intersect_modflow_cell
        
def recharge_cell_define(modflow_cell_grid, dis):
    recharge_cell = {}
    for i in range(modflow_cell_grid.shape[0]):
        row, col = modflow_cell_grid.loc[i,"row"], modflow_cell_grid.loc[i,"col"]
        top =  dis.top.array[row-1][col-1]
        bottom = dis.botm.array[0][row-1][col-1]
        values = np.arange(top - 0.5, bottom, -1)
        data = np.zeros(len(values))
        recharge_cell[(row,col)]=pd.DataFrame(data, index=values, columns=["SW"])
    return recharge_cell
    
def year_read(TXTINOUT):
    import re
    with open(os.path.join(TXTINOUT, "file.cio")) as fio:
        lines = fio.readlines()
        for line in lines:
            if "IYR" in line:
                field=line.split(" ")
                field = [re.sub(r'\s+', '', item) for item in field if item.strip()]
                st_year=int(field[0])
            elif "NYSKIP" in line:
                field=line.split(" ")
                field = [re.sub(r'\s+', '', item) for item in field if item.strip()]
                st_year+=int(field[0])
                break
    return st_year

def swat_hec_manning(TXTINOUT,HecData, version):
    rte_file = glob.glob(os.path.join(TXTINOUT, "*.rte"))
    rte_file_names = [os.path.basename(file) for file in rte_file]
    N_value={}
    for idx, rte_name in enumerate(rte_file_names):
        with open(rte_file[idx], 'r') as rte:
            lines = rte.readlines()
            subbasin_numbers = int(re.findall(r'Subbasin:\s*(\d+)', lines[0])[0])
            rte_manip = swatpy.rteManipulator(rte_name, ["CH_N2"], TXTINOUT)
            row, col1, col2, dig = rte_manip.parInfo["CH_N2"]
            N_v = float(rte_manip.textOld[row - 1][col1:col2])
            N_value[subbasin_numbers] = N_v
            
    ras = Ras(HecData + ".prj" , version)
    api = API(ras)
    for river in api.ras.rivers:
        river_name = river.river
        for reach in river.reaches:
            reach_name = reach.reach
            apply_N = N_value[int(reach_name)]
            api.params.modifyN(apply_N, river=river_name, reach=reach_name)

def main_translate(HecData, Python_need_shp_folder, modflow_model_name, ModData, exe_path, sub_program_route, SwatData, initial_head, Modflow_work_space, python_code_path, river_conductivity, river_thickness, output_data_space, pest_dir, Hec_name, re_write, Td_area_name):
    pkl_path = os.path.join(sub_program_route, "run_par.pkl")
    
    if os.path.exists(pkl_path) and re_write == False:
        with open(os.path.join(sub_program_route, "run_par.pkl"), 'rb') as pkl_file:
            run_hsm_params = pickle.load(pkl_file)
            
            TXTINOUT = run_hsm_params["TXTINOUT"]
            SwatData = run_hsm_params["SwatData"]
            flow_write_dic = run_hsm_params["flow_write_dic"]
            hec_route = run_hsm_params["hec_route"]
            version = run_hsm_params["version"]
            Flow_file = run_hsm_params["Flow_file"]
            sub_program_route = run_hsm_params["sub_program_route"]
            Plan_file = run_hsm_params["Plan_file"]
            resolution = run_hsm_params["resolution"]
            output_data_space = run_hsm_params["output_data_space"]
            wel_time_dic = run_hsm_params["wel_time_dic"]
            river_conductivity = run_hsm_params["river_conductivity"]
            river_thickness = run_hsm_params["river_thickness"]
            modflow_swat_hru = run_hsm_params["modflow_swat_hru"]
            rch = run_hsm_params["rch"]
            cols_flat = run_hsm_params["cols_flat"]
            rows_flat = run_hsm_params["rows_flat"]
            model_layer = run_hsm_params["model_layer"]
            intersect_modflow_cell = run_hsm_params["intersect_modflow_cell"]
            modflow_cell_grid = run_hsm_params["modflow_cell_grid"]  # GeoDataFrame or array
            Dem_df = run_hsm_params["Dem_df"]  # DataFrame
            st_year = run_hsm_params["st_year"]
            pest_dir = run_hsm_params["pest_dir"]
            Hec_name = run_hsm_params["Hec_name"]
            hy_dic = run_hsm_params["hy_dic"]
            Python_need_shp_folder = run_hsm_params["Python_need_shp_folder"]
            python_code_path = run_hsm_params["python_code_path"]
            end_day = run_hsm_params["end_day"]
            s_polygon_gdf = run_hsm_params["s_polygon_gdf"]
            
            mf, wel, wel_time_dic, bas_dic, model_col,xll, xul, yll, yul, dis, bas, rch, model_col, model_row, model_layer, hy_dic = Modflow_read(modflow_model_name, ModData, exe_path, initial_head, Modflow_work_space)
            
            bas_dic = run_hsm_params["bas_dic"]
            wel = run_hsm_params["wel"]
            mf = run_hsm_params["mf"]  # MODFLOW object

    else:
        Geom_file, Flow_file, version, formatted_version, Plan_file, end_day = Hec_Data_Read(HecData)
        shp_route , origin_shape_gdf, s_polygon_gdf= shape_file_read(Python_need_shp_folder, HecData, Plan_file, Td_area_name)
        Dem_df, crs, resolution = raster_dem_read(Python_need_shp_folder, origin_shape_gdf)
        mf, wel, wel_time_dic, bas_dic, model_col,xll, xul, yll, yul, dis, bas, rch, model_col, model_row, model_layer, hy_dic = Modflow_read(modflow_model_name, ModData, exe_path, initial_head, Modflow_work_space)
    
        # 두 번째 작업들
        flow_write_dic = Hec_river_write(Flow_file, HecData)
        modflow_cell_grid, modflow_swat_hru = modflow_swat_linker(shp_route, xll, yll, xul, yul, bas_dic, dis, model_col, sub_program_route, crs)
        intersect_modflow_cell = modcell_basin(modflow_cell_grid, shp_route)
        recharge_cell = recharge_cell_define(modflow_cell_grid, dis)
        
        TXTINOUT_origin = os.path.join(SwatData, r"Scenarios\Default\TxtInOut")
        TXTINOUT = os.path.join(SwatData, r"Scenarios\Default\TxtInOut2")
        if not os.path.exists(TXTINOUT):
            shutil.copytree(TXTINOUT_origin, TXTINOUT)
        
        st_year = year_read(TXTINOUT)
        unique_hru_ids = modflow_swat_hru[r'GRIDCODE'].unique()
        output_txt=os.path.join(TXTINOUT, "GW_zero_HRU.txt")
        with open(output_txt, 'w') as f:
            f.write(f"{len(unique_hru_ids)}\n")
            for hru_id in unique_hru_ids:
                f.write(f"{hru_id}\n")  # 각 값을 한 행으로 작성
        
        recharge = rch.rech.array[0][0]
        recharge[:] = 0
        rch = recharge
        modflow_swat_hru = {grid_id: df for grid_id, df in modflow_swat_hru.groupby('grid_id')}
    
        rows, cols = np.indices((model_row, model_col))
        rows = rows+1
        cols = cols+1
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        
        hec_route = HecData
        
        run_hsm_params = {
            "TXTINOUT": TXTINOUT,
            "SwatData": SwatData,   
            "flow_write_dic": flow_write_dic,
            "hec_route": HecData,
            "version": version,
            "Flow_file": Flow_file,
            "sub_program_route": sub_program_route,
            "Plan_file": Plan_file,
            "resolution": resolution ,
            "output_data_space": output_data_space ,
            "wel_time_dic": wel_time_dic,
            "bas_dic": bas_dic ,
            "wel": wel ,
            "mf": mf,
            "river_conductivity": river_conductivity ,
            "river_thickness": river_thickness ,
            "modflow_swat_hru": modflow_swat_hru ,
            "rch": rch,
            "cols_flat": cols_flat,
            "rows_flat": rows_flat ,
            "model_layer": model_layer ,
            "intersect_modflow_cell": intersect_modflow_cell,
            "modflow_cell_grid": modflow_cell_grid,  ##
            "Dem_df": Dem_df,
            "st_year" : st_year,
            "pest_dir" : pest_dir,
            "Hec_name" : Hec_name,
            "hy_dic" : hy_dic,
            "Python_need_shp_folder" : Python_need_shp_folder,
            "python_code_path" : python_code_path,
            "end_day" : end_day,
            "s_polygon_gdf" : s_polygon_gdf}
    
        with open(os.path.join(sub_program_route, "run_par.pkl"), 'wb') as pkl_file:
            pickle.dump(run_hsm_params, pkl_file)
        
    return TXTINOUT, SwatData, flow_write_dic, hec_route, version, Flow_file, sub_program_route, Plan_file, resolution, output_data_space, \
        bas_dic, wel, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, cols_flat, rows_flat, model_layer, intersect_modflow_cell, modflow_cell_grid, Dem_df, st_year, Hec_name, end_day, s_polygon_gdf
                