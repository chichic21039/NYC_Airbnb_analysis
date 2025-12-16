import numpy as np
import os
def parse_csv(file_path):
    rows = []
    inside_quotes = False
    current_field = []
    current_row = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    i = 0
    while i < len(data):
        char = data[i]

        if char == '"':
            # "" -> escaped quote
            if inside_quotes and i + 1 < len(data) and data[i+1] == '"':
                current_field.append('"')
                i += 1
            else:
                inside_quotes = not inside_quotes

        elif char == ',' and not inside_quotes:
            # kết thúc field
            current_row.append(''.join(current_field))
            current_field = []

        elif char == '\n' and not inside_quotes:
            # kết thúc dòng
            current_row.append(''.join(current_field))
            current_field = []

            rows.append(current_row)
            current_row = []

        else:
            # nếu inside quotes và gặp newline
            if char == '\n' and inside_quotes:
                current_field.append(" ")
            else:
                current_field.append(char)

        i += 1

    # Xử lý dòng cuối nếu không kết thúc bằng \n
    if current_field or current_row:
        current_row.append(''.join(current_field))
        rows.append(current_row)

    # Lấy header
    header = rows[0]
    num_cols = len(header)

    # Bổ sung cột trống cho các dòng thiếu
    cleaned_rows = []
    for row in rows[1:]:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        cleaned_rows.append(row)

    return header, cleaned_rows

def load_airbnb(filename="AB_NYC_2019.csv", source_dir="raw"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    path = os.path.join(project_root, "data", source_dir, filename)
    header, rows = parse_csv(path)
    return header, rows
def to_float_safe(col_values):
    """
    Convert a list of strings to float array; non-convertible -> np.nan
    """
    out = np.empty(len(col_values), dtype=float)
    for i, v in enumerate(col_values):
        if v is None:
            out[i] = np.nan
            continue
        s = str(v).strip()
        if s == "":
            out[i] = np.nan
            continue
        try:
            out[i] = float(s)
        except Exception:
            # try remove common thousands separators and currency symbols
            s2 = s.replace(",", "").replace("$", "").replace("€", "")
            try:
                out[i] = float(s2)
            except Exception:
                out[i] = np.nan
    return out

def columns_from_rows(rows):
    """
    rows: list of rows (each row is list of fields), assumed same length as header
    returns: list of column lists (strings)
    """
    if len(rows) == 0:
        return []
    ncols = max(len(r) for r in rows)
    cols = [[] for _ in range(ncols)]
    for r in rows:
        # pad if shorter
        for j in range(ncols):
            if j < len(r):
                cols[j].append(r[j])
            else:
                cols[j].append("")
    return cols

def convert_columns(rows, header, numeric_cols=None):
    """
    Convert selected columns to numeric arrays (float), others kept as list of strings.
    Inputs:
        rows: list of rows (list of lists)
        header: list of column names
        numeric_cols: list of column indices OR list of column names to convert; if None -> auto detect
    Returns:
        cols_dict: {colname: {"values": array or list, "is_numeric": bool}}
    """
    cols = columns_from_rows(rows)
    ncols = len(cols)
    colnames = header.copy()
    # build numeric_cols indices if passed as names
    if numeric_cols is None:
        # auto-detect: try to convert, if >50% convertible -> numeric
        numeric_cols_idx = []
        for j, col in enumerate(cols):
            arr = to_float_safe(col)
            non_nan = np.count_nonzero(~np.isnan(arr))
            if non_nan / max(1, len(arr)) >= 0.5:
                numeric_cols_idx.append(j)
    else:
        # convert numeric_cols (accept names or indices)
        numeric_cols_idx = []
        for x in numeric_cols:
            if isinstance(x, int):
                numeric_cols_idx.append(x)
            else:
                if x in colnames:
                    numeric_cols_idx.append(colnames.index(x))

    cols_dict = {}
    for j, name in enumerate(colnames):
        if j in numeric_cols_idx:
            cols_dict[name] = {"values": to_float_safe(cols[j]), "is_numeric": True, "index": j}
        else:
            cols_dict[name] = {"values": np.array(cols[j], dtype=object), "is_numeric": False, "index": j}
    return cols_dict

def save_cols_dict_to_csv(cols_dict, header, save_path):
    """
    Ghi dữ liệu từ cols_dict vào file CSV với định dạng đúng chuẩn.
    (Hàm này sao chép logic ghi file từ 02_preprocessing.ipynb)
    """
    
    # Tạo thư mục nếu chưa tồn tại
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_cols = header
    if not all_cols and cols_dict:
        all_cols = list(cols_dict.keys())
        
    if not all_cols:
        print("Lỗi: Không có cột để ghi.")
        return

    num_rows = len(cols_dict[all_cols[0]]["values"])

    print("--- BẮT ĐẦU GHI FILE VỚI ĐỊNH DẠNG CSV CHUẨN (CÓ DẤU NGOẶC KÉP) ---")

    with open(save_path, 'w', encoding='utf-8', newline='') as f: # Dùng newline='' cho tính tương thích
        # 1. Ghi Header
        f.write(','.join(all_cols) + '\n') 
        
        # 2. Ghi từng dòng
        for i in range(num_rows):
            row_values = []
            for name in all_cols:
                value = cols_dict[name]["values"][i]
                
                # Logic định dạng giá trị (như trong code gốc của bạn)
                if cols_dict[name]["is_numeric"] and name not in ["id", "host_id"]:
                    if name in ['id', 'host_id', 'minimum_nights', 'number_of_reviews']:
                        formatted_val = str(int(value)) 
                    else:
                        formatted_val = f"{value:.4f}" 
                    row_values.append(formatted_val)
                else:
                    str_value = str(value) 
                    
                    if ',' in str_value or '\"' in str_value:
                        str_value = str_value.replace('\"', '\"\"')
                        row_values.append(f'"{str_value}"')
                    else:
                        row_values.append(str_value)
                        
            f.write(','.join(row_values) + '\n')
            
    print(f"\n--- HOÀN TẤT GHI FILE TẠI {save_path} ---")

def create_grid_id(latitude, longitude, grid_size=0.01):
    """Tạo Grid ID với kích thước lưới tùy chỉnh (0.01 hoặc 0.005)"""
    if latitude is None or longitude is None:
        return None
    # Công thức lát gạch (tiling)
    lat_bin = round(np.floor(float(latitude) / grid_size) * grid_size, 5)
    lon_bin = round(np.floor(float(longitude) / grid_size) * grid_size, 5)
    return f"{lat_bin}_{lon_bin}"

def add_grid_id_column(cols_dict, header, grid_size, col_name='grid_id'):
    """Thêm cột Grid ID vào cols_dict."""
    lat_values = cols_dict['latitude']['values']
    lon_values = cols_dict['longitude']['values']
    grid_ids = [
        create_grid_id(lat, lon, grid_size) 
        for lat, lon in zip(lat_values, lon_values)
    ]
    cols_dict[col_name] = {'values': np.array(grid_ids, dtype=object), 'is_numeric': False}
    if col_name not in header: header.append(col_name)
    return cols_dict, header

def add_price_per_night_proxy(cols_dict, header):
    """Tính toán Price per Night Proxy (Feature Phụ trợ)."""
    price = np.array(cols_dict['price']['values'])
    min_nights = np.array(cols_dict['minimum_nights']['values'])
    
    # Tránh chia cho 0 và NaN
    valid_mask = (min_nights > 0) & ~np.isnan(price)
    
    # Tạo một mảng rỗng hoặc None
    price_proxy = np.full(len(price), np.nan, dtype=np.float64)
    price_proxy[valid_mask] = price[valid_mask] / min_nights[valid_mask]
    
    feature_name = 'price_per_night_proxy'
    cols_dict[feature_name] = {'values': price_proxy, 'is_numeric': True}
    if feature_name not in header: header.append(feature_name)
    return cols_dict, header


def calculate_aggregation(cols_dict, group_col, target_cols):
    """
    Thực hiện Groupby và tính toán các chỉ số thống kê (STD, Median, Count).
    Returns: {grid_id: {'std_price': X, 'median_price': Y, ...}, ...}
    """
    group_ids = cols_dict[group_col]['values']
    
    # Nhóm dữ liệu
    grouped_data = {}
    for i, group_id in enumerate(group_ids):
        if group_id is None: continue
        if group_id not in grouped_data:
            grouped_data[group_id] = {col: [] for col in target_cols}
            
        for col in target_cols:
            val = cols_dict[col]['values'][i]
            if val is not None and not np.isnan(val):
                grouped_data[group_id][col].append(val)

    # Tính toán thống kê
    agg_results = {}
    for group_id, data in grouped_data.items():
        results = {'count': len(data[target_cols[0]])}
        
        for col in target_cols:
            arr = np.array(data[col], dtype=np.float64)
            if len(arr) > 1:
                results[f'median_{col}'] = np.median(arr)
                results[f'std_{col}'] = np.std(arr)
            elif len(arr) == 1:
                results[f'median_{col}'] = arr[0]
                results[f'std_{col}'] = 0.0 # STD = 0 nếu chỉ có 1 điểm
            else:
                results[f'median_{col}'] = np.nan
                results[f'std_{col}'] = np.nan
        agg_results[group_id] = results

    return agg_results

def map_features_to_cols_dict(cols_dict, header, agg_results, map_col, feature_names):
    """
    Ánh xạ các kết quả aggregation (agg_results) trở lại từng dòng listing.
    """
    map_ids = cols_dict[map_col]['values']
    
    # Khởi tạo các list để lưu giá trị của cột mới
    feature_values = {f: [] for f in feature_names}
    
    for map_id in map_ids:
        if map_id in agg_results:
            agg_data = agg_results[map_id]
            for feature in feature_names:
                # Dùng .get(feature, np.nan) để xử lý ID bị thiếu
                feature_values[feature].append(agg_data.get(feature, np.nan)) 
        else:
            for feature in feature_names:
                feature_values[feature].append(np.nan)

    # Cập nhật cols_dict và header
    for feature in feature_names:
        cols_dict[feature] = {'values': np.array(feature_values[feature], dtype=np.float64), 'is_numeric': True}
        if feature not in header:
            header.append(feature)
            
    return cols_dict, header