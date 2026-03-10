import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_and_clean(data_dir):
    print(f"Loading participant CSV files from {data_dir}...")
    all_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    df_list = []

    for file in all_files:
        try:
            if any(name in file for name in ["target_and_lures.csv", "abruptmovies.csv", "naturalmovies.csv", "Demographic data"]):
                continue

            temp_df = pd.read_csv(file)
            if 'participant' in temp_df.columns and len(temp_df) > 0:
                df_list.append((temp_df, file))
        except Exception:
            pass

    if not df_list:
        print("\nERROR: No participant data CSV files found containing a 'participant' column!")
        return None, None

    part_files = {}
    just_dfs = []

    for temp_df, file_path in df_list:
        orig_part = str(temp_df['participant'].iloc[0])
        if 'orig_participant' not in temp_df.columns:
            temp_df['orig_participant'] = orig_part

        if orig_part not in part_files:
            part_files[orig_part] = []
            part = orig_part
        else:
            part = f"{orig_part}_dup{len(part_files[orig_part])}"
            temp_df['participant'] = part

        part_files[orig_part].append(os.path.basename(file_path))
        just_dfs.append(temp_df)

    df = pd.concat(just_dfs, ignore_index=True)
    print(f"\nSuccessfully merged {len(just_dfs)} participant data files.")

    duplicates = {part: files for part, files in part_files.items() if len(files) > 1}

    df['Group'] = df['participant'].astype(str).apply(
        lambda x: 'AB' if 'AB' in x else ('NB' if 'NB' in x else np.nan)
    )

    def extract_sub_num(participant_str):
        if not isinstance(participant_str, str): return np.nan
        import re
        match = re.search(r'sub(\d+)', participant_str.lower())
        if match: return int(match.group(1))
        return np.nan

    df['sub_num'] = df['participant'].apply(extract_sub_num)

    df = df[(df['sub_num'] > 13) | (df['sub_num'].isna())].copy()

    def get_frame_type(img_path):
        if not isinstance(img_path, str): return np.nan
        if '_BB_' in img_path: return 'BB'
        elif '_EM_' in img_path: return 'EM'
        return np.nan

    df['Frame_Type'] = df['target_img'].apply(get_frame_type)

    corr_col = 'resp.corr' if 'resp.corr' in df.columns else 'recogloop.resp.corr'
    rt_col = 'resp.rt' if 'resp.rt' in df.columns else 'recogloop.resp.rt'
    conf_col = 'conf_radio.response' if 'conf_radio.response' in df.columns else 'recogloop.conf_radio.response'

    for col in [corr_col, rt_col, conf_col]:
        if col not in df.columns:
            df[col] = np.nan

    recog_df = df.dropna(subset=[corr_col]).copy()

    for col in [corr_col, rt_col, conf_col]:
        recog_df[col] = recog_df[col].astype(str).str.replace(r'\[|\]|None', '', regex=True)
        recog_df[col] = pd.to_numeric(recog_df[col], errors='coerce')

    demo_files = glob.glob(os.path.join(data_dir, "**", "Demographic data*.xlsx"), recursive=True)
    if demo_files:
        try:
            demo_df = pd.read_excel(demo_files[0])
            print("\n=== Demographics Summary ===")
            print(demo_df.describe(include='all'))
            if 'Sub ID' in demo_df.columns:
                demo_df.rename(columns={'Sub ID': 'orig_participant'}, inplace=True)
            elif 'participant' in demo_df.columns:
                demo_df.rename(columns={'participant': 'orig_participant'}, inplace=True)

            if 'orig_participant' in demo_df.columns and 'orig_participant' in recog_df.columns:
                recog_df = recog_df.merge(demo_df, on='orig_participant', how='left')
            else:
                recog_df = recog_df.merge(demo_df, on='participant', how='left')
        except ImportError:
            print("\n[NOTE] Could not read demographics: please run 'pip install openpyxl'")
        except Exception as e:
            print(f"\nCould not process demographic data: {e}")

    print("\n=== Calculating Vigilance Check ===")
    vigilance_data = []

    for part, group in df.groupby('participant'):
        inst_col = 'instruction_2.stopped' if 'instruction_2.stopped' in group.columns else None
        vid_col = 'Videos.stopped' if 'Videos.stopped' in group.columns else None

        inst_stop = group[inst_col].dropna().min() if inst_col else np.nan
        vid_stop = group[vid_col].dropna().max() if vid_col else np.nan
        encoding_time = vid_stop - inst_stop if pd.notnull(inst_stop) and pd.notnull(vid_stop) else np.nan

        is_vigilant = encoding_time <= 1623 if pd.notnull(encoding_time) else True
        vigilance_data.append({'participant': part, 'encoding_time_sec': encoding_time, 'is_vigilant': is_vigilant})

    vig_df = pd.DataFrame(vigilance_data)
    excluded_vig = vig_df[~vig_df['is_vigilant']]['participant'].tolist()
    print(f"Excluded participants due to vigilance ({len(excluded_vig)}): {excluded_vig}")

    recog_df = recog_df.merge(vig_df[['participant', 'is_vigilant']], on='participant', how='left')
    clean_df = recog_df[recog_df['is_vigilant'] == True].copy()

    clean_df = clean_df[(clean_df[rt_col] > 0.2) & clean_df[rt_col].notnull()]

    if len(clean_df) == 0:
        print("\nERROR: No valid recognition trials remaining after cleaning!")
        return None, None

    final_subs = clean_df['participant'].nunique()
    total_files = len(df_list)
    initial_subs = df['participant'].nunique()

    subs_after_13_exclusion = df[(df['sub_num'] > 13) | (df['sub_num'].isna())]['participant'].nunique()
    first_13_count = initial_subs - subs_after_13_exclusion

    subs_after_no_trials = recog_df['participant'].nunique()
    empty_trials_count = subs_after_13_exclusion - subs_after_no_trials

    subs_after_vigilance = recog_df[recog_df['is_vigilant'] == True]['participant'].nunique()

    subs_after_rt = clean_df['participant'].nunique()
    fast_rt_drops = subs_after_vigilance - subs_after_rt

    print(f"\n=== Cleaning Summary ===")
    print(f"Total participant CSV files loaded: {total_files}")
    print(f"Total Unique Participants identified initially: {initial_subs}")
    print(f"  - Excluded because ID <= 13 (corrupt blocks): {first_13_count}")
    print(f"  - Excluded because they had NO valid recognition trials recorded: {empty_trials_count}")
    print(f"  - Excluded due to Vigilance Check: {len(excluded_vig)}")
    print(f"  - Excluded because ALL their valid trials were <0.2s: {fast_rt_drops}")
    print(f"Total valid participants remaining for analysis: {final_subs}")
    print(f"Total Subjects Excluded Overall: {initial_subs - final_subs}")

    col_names = {'corr_col': corr_col, 'rt_col': rt_col, 'conf_col': conf_col}
    return clean_df, col_names
