import os
import pandas as pd
import decord
from tqdm import tqdm

def assign_clip_id(group):
    group = group.reset_index(drop=True)  # Reset index within the group
    group['clip_id'] = group['video_id'] + '_' + group.index.astype(str)
    return group

def process_videos(data_dir):
    all_metadata = []
    
    # Iterate over all participant directories
    for pid in os.listdir(data_dir):
        pid_dir = os.path.join(data_dir, pid)
        if os.path.isdir(pid_dir):
            # Process each video in the participant's directory
            video_files = os.listdir(pid_dir)
            for video_file in tqdm(video_files, desc=f"Processing videos in {pid_dir}"):
                if video_file.endswith('.MP4'):
                    video_id = os.path.splitext(video_file)[0]
                    full_video_fp = os.path.join(pid_dir, video_file)
                    try:
                        vr = decord.VideoReader(full_video_fp)
                        fps = vr.get_avg_fps()
                        if fps != 50.0:
                            print(video_file)
                            print(fps)
                        frames_per_second = int(round(fps))
                        half_second_frames = int(round(fps / 2))
                        start_frames = []
                        end_frames = []
                        current_frame = 0
                        while current_frame + frames_per_second <= len(vr):
                            start_frames.append(current_frame)
                            end_frames.append(current_frame + frames_per_second - 1)
                            current_frame += frames_per_second - half_second_frames  # Move by 1 second minus half second for overlap
                        
                        # Create a DataFrame for the current video
                        metadata = pd.DataFrame({
                            'start_frame': start_frames,
                            'end_frame': end_frames,
                            'video_id': video_id
                        })
                        all_metadata.append(metadata)
                    except Exception as e:
                        print(f"Failed to process {video_file}: {e}")
    
    # Concatenate all metadata into a single DataFrame
    if all_metadata:
        final_metadata = pd.concat(all_metadata, ignore_index=True)
        final_metadata = final_metadata.groupby('video_id').apply(assign_clip_id).reset_index(drop=True)
        return final_metadata
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no videos were processed

# Usage
data_dir = '/datasets01/EPIC-KITCHENS-100-VIDEOS-ht256px/060122/'
result_df = process_videos(data_dir)

output_file_path = "/private/home/arjunrs1/epic-kitchens-100-annotations/"
output_file_name = "EPIC_100_EgoVLP_feature_timestamps_all_splits.csv"
full_output_path = os.path.join(output_file_path, output_file_name)
#result_df.to_csv(full_output_path, index=False)
#print(f"Data saved to {full_output_path}")
