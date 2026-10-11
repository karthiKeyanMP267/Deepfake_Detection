import pandas as pd
import glob

manifests = glob.glob('**/*manifest*.csv', recursive=True)
for m in manifests:
    try:
        df = pd.read_csv(m)
        if 'sample_id' in df.columns and 'sample_00014' in df['sample_id'].values:
            row = df[df['sample_id'] == 'sample_00014'].iloc[0]
            print(f"{m}:")
            print("  video_path:", row.get("video_path", "N/A"))
            print("  label:", row.get("label", "N/A"), "manipulation:", row.get("manipulation", "N/A"))
    except Exception as e:
        pass
