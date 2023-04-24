import os
import librosa
import math
import json
import soundfile as sf

DATASET_PATH = "genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionaty to store data
    data = {
        "mapping": ["classical", "blues"], # labels
        "mfcc": [], # training data
        "labels": [] # training labels
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # num_samples_per_segment = int(SAMPLE_RATE / num_segments)
    
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 


    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/") # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1] # the last blues
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                
                try:
                    signal, sr = sf.read(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    continue

                # Process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s current segment
                    finish_sample = start_sample + num_samples_per_segment
                    y = signal[start_sample:finish_sample]
                    
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)        
                        print("{}, segment:{}".format(file_path, s))                
    
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4) # data is the dictionary, fp is the file path
    
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
