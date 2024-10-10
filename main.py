python_code_path = r"C:\Users\user\Desktop\SWAT_MODFLOW\Python"

import os
os.chdir(python_code_path)

from HSM_translate import main_translate
from HSM_run import run_HSM
from HSM_out import export_out_data, GW_head_obs_data, river_obs_data, zone_budget_swat, river_point_in_out, modflow_zone_budget_clear
from HSM_pest import make_pest_file, pest_run

ModData = r"C:\Users\user\Desktop\SWAT_MODFLOW\1_modflow"                #modflow 루트
HecData = r"C:\Users\user\Desktop\SWAT_MODFLOW\2_hec_test"   #hecras 루트
SwatData = r"C:\Users\user\Desktop\SWAT_MODFLOW\3_swat"   #swat 루트  TXTINOUT 파일은 복사해서 TXTINOUT2 폴더로 만들도록
modflow_model_name = "ts_main"   # modflow 모델 이름
Hec_name="hec_641_t2"
Td_area_name = "Perimeter 1"

Modflow_work_space = r"C:\Users\user\Desktop\SWAT_MODFLOW\4_linked_modflow"  #modflow 결과 작업파일
output_data_space = r"C:\Users\user\Desktop\SWAT_MODFLOW\5_outputdata"      #결과 생성 파일

Python_need_shp_folder = r"C:\Users\user\Desktop\SWAT_MODFLOW\0_gis\Python_need_shp"    #필요 shp 파일 루트 river hru watershed    (river는 hecras 데이터 생성용으로 만들어낸 river가 필요, 나머지는 swat 결과)
exe_path = os.path.join(Modflow_work_space, "MODFLOW-NWT.exe")          #엔진 경로
sub_program_route = r"C:\Users\user\Desktop\SWAT_MODFLOW\Python\subprogram" #대충 아무거나 PYTHON이 필요한 SHP 파일 저장해둘 공간 아무곳
initial_head=r"C:\Users\user\Desktop\SWAT_MODFLOW\F1.HDS" #초기수두          # 초기 수두 hds

pest_dir = r"C:\Users\user\Desktop\SWAT_MODFLOW\6_Pest"
pest_need_file = r"C:\Users\user\Desktop\SWAT_MODFLOW\pest_need"

river_thickness = 1             # 하상퇴적층 두께
river_conductivity = 50         # 하상퇴적층 수리전도도

re_write = True
 
# 모델 데이터 변환----------------------------------------------------`-----------------------------------------------------------------------------------
TXTINOUT, SwatData, flow_write_dic, hec_route, version, Flow_file, sub_program_route, Plan_file, resolution, output_data_space, \
    bas_dic, wel, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, cols_flat, rows_flat, model_layer, intersect_modflow_cell, modflow_cell_grid, Dem_df, st_year, Hec_name, end_day, s_polygon_gdf \
    = main_translate(HecData+"/"+Hec_name, Python_need_shp_folder, modflow_model_name, ModData, exe_path, sub_program_route, SwatData, initial_head, Modflow_work_space, python_code_path, river_conductivity, river_thickness, output_data_space, pest_dir, Hec_name, re_write, Td_area_name)

# 모델 실행---------------------------------------------------------------------------------------------------------------------------------------------
run_HSM(TXTINOUT, write_result, hec_route, version, Flow_file, sub_program_route, Plan_file, river_gdfs, \
        river_horizontal_distance, Dem_df, resolution, modflow_cell_grid, swat_hec_link_dic, output_data_space, final_interpolated_gdf, \
        wel_time_dic, bas_dic, wel, mf, river_conductivity, river_thickness, modflow_swat_hru, rch, cols_flat, rows_flat, model_layer, weir_dic, \
        intersect_modflow_cell, drain_out_weight, nearest_river_weir, basin_to_river)

# 모델 결과 뽑기----------------------------------------------------------------------------------------------------------------------------------------
export_out_data(sub_program_route, output_data_space, st_year)     #gis 격자별 데이터 뽑기
GW_head_obs_data(output_data_space, st_year, Modflow_work_space)   # 관측정 데이터 뽑기
river_obs_data(output_data_space, st_year, Python_need_shp_folder) # 강 관측정 유량 수위 데이터 뽑기
zone_budget_swat(TXTINOUT, st_year, output_data_space)             # hru 결과 뽑기 swat에서
river_point_in_out(st_year, output_data_space, sub_program_route)
modflow_zone_budget_clear(st_year, output_data_space)
#Pest data-----------------------------------------------------------------------------------------------------------------------------------------------

pest_st_year=2023
pest_end_year=2024
simulated_year_start_day = 1
simulated_year_end_day=200
pest_output, pest_mf = make_pest_file(pest_dir, SwatData, output_data_space, st_year, Modflow_work_space, HecData, Python_need_shp_folder, pest_end_year, pest_st_year, mf, sub_program_route, simulated_year_end_day, python_code_path, pest_need_file, simulated_year_start_day)
pest_run(pest_dir)