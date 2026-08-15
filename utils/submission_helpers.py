from utils.submission_imports import *

def normalize_targets(coords, mins, maxs):
    """Maps raw cm coordinates to a [-1, 1] range."""
    # Scale to [0, 1] first, then shift to [-1, 1]
    return 2.0 * (coords - mins) / (maxs - mins) - 1.0

def denormalize_predictions(preds, mins, maxs):
    """Maps [-1, 1] model outputs back to raw cm coordinates."""
    # Reverse shift from [-1, 1] to [0, 1], then scale back to cm
    return (preds + 1.0) / 2.0 * (maxs - mins) + mins


@staticmethod
def audio_simulator(input, source_loc:tuple, mic_locs:list|tuple):
    """
    Def:
    :params input
    :params tuple of floats source_loc: Holding the cartesian coordinates of the source; 
    :params list of tuples mic_locs: Holding the cartesian coordinates of multiple microphones. If all the microphones share the same location then the argument becomes a tuple

    Return: waveforms (visualiazation) and time_delays
    """

    def _time_delay_calculation(mic=mic_locs, source=source_loc, common_loc=False):
        for locations in mic:
            relative_time_delays.append(np.round((np.sqrt( (source[0]-locations[0])**2 + (source[1]-locations[1])**2)/c)*sr))

        # 2. Locate the furthest situated microphone by time_delay: the more samples, the furthest
        furtherst = max(relative_time_delays)
        for relative_time in relative_time_delays:
            delay = furtherst-relative_time
            waveforms.append(input[int(delay):])
    
    c:int=340*100
    sr:int = 48000
    relative_time_delays:list = []
    time_delay_of_arival:int = 0
    waveforms:list = []

    if isinstance(mic_locs, tuple):
        _time_delay_calculation(True)
    elif isinstance(mic_locs, list):
        _time_delay_calculation()
    
    
    return waveforms, relative_time_delays


class CustomAudioDecoderData(Dataset):
    """
    :param float t:
    :param list of integers mic_locs:
    :param tuple holding float values n_sources_min_max:
    ....
    """
    def __init__(self, t, mic_locs, n_sources_min, n_sources_max, 
                 x_loc_min, x_loc_max, y_loc_min, y_loc_max, 
                 split='train', epoch_len:int=2560,
                 data_folder=[r'utils\assignment_2_dataset\train',
                      r'utils\assignment_2_dataset\validation',
                      r'utils\assignment_2_dataset\test'], transforms:bool = None, random_assignment:bool=True, dynamic:bool=True):

        # 1.1 Fail save, ensuring NO-CAPITAL charachters are accidentally inserted;        
        self.split = split.lower()
        # 1.2 Map split string to the correct index of data_folder (0=train, 1=val, 2=test)
        split_mapping = {'train': 0, 'val': 1, 'test': 2}
        if self.split not in split_mapping:
            raise ValueError("split must be 'train', 'val', or 'test'")
        # 1.3 Set a value (e.g., 0, 1 or 2) based on the subset of interest specified at class instantiation time;
        self.split_idx = split_mapping[self.split]

        self.dataset_root_paths = data_folder
        self.transform = transforms
        self.end_time = t/1000 
        self.mic_locs = mic_locs
        self.n_sources = (n_sources_min, n_sources_max)
        self.x_y_min_max = (x_loc_min, x_loc_max, y_loc_min, y_loc_max)
        self.rdm_assign = random_assignment
        self.dynamic = dynamic
        self.epoch_len = epoch_len

        # 1. Collecting data paths
        all_data_paths = self._retrieve_and_assemble_data_paths()

        # 2. Declare active file paths based on step `1.3`
        self.active_paths = all_data_paths[self.split_idx]
    
    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):

        # 3.1 Determine how many sources;
        n_sources = rdm.randint(self.n_sources[0], self.n_sources[1])
        # 3.2 Create data holders aka containers;
        all_simulated_waveforms, source_locations = [], []
        # 4. Loop over sources
        for _ in range(n_sources):
            # 4.1 Target a single file path string using the index [1] and a randomly chosen index [rdm]
            file_path = rdm.choice(self.active_paths)
            # 4.2 Assign and save/append the randomly generated coordinates of the sources
            if self.rdm_assign:
                x, y = rdm.uniform(self.x_y_min_max[0], self.x_y_min_max[1]), rdm.uniform(self.x_y_min_max[2], self.x_y_min_max[3])
            else:
                x = self.x_y_min_max[0]
                y = self.x_y_min_max[2]

            source_locations.append((x,y))
            # 4.3 Safely instantiate AudioDecoder with a single string path [1]
            decoder = AudioDecoder(file_path)
            if self.transform:
                # 4.4 Extract audio tensor stream and cut adequately
                waveform = decoder.get_samples_played_in_range(start_seconds=0, stop_seconds=self.end_time) # Length of the returned sample
                #TODO Implement if statement to check for the number of channels
                mono = torch.mean(waveform.data, dim=0) # Averaging the waveform holding at least two channels
                waveforms, delays = audio_simulator(mono, source_loc=(x, y), mic_locs=self.mic_locs) 
                # 4.4.1 Find the samllest length in the list of waveforms
                # ... then use it as a refference for a clamping threshold
                min_length = min([w.shape[0] for w in waveforms])
                # 4.4.2 Trim every waveform to match that minimum length
                trimmed_waveforms = [w[:min_length] for w in waveforms] # Shape waveforms:  [torch.tensor(simmulated waveform) for wave in range(no. mics)]
                # 4.4.3 Stack the trimmed waveforms                     # 
                source_tensor = torch.stack(trimmed_waveforms) # Shape: [n_mics, samples]
                all_simulated_waveforms.append(source_tensor)
                # # Debug statement 2
                # print("Waveform-decoder shape {}\n"
                # "Waveforms-decoder after mono p[ shape {}\n" \
                # "Waveforms after audio simulator and trimming {} plus size of each element {}\n" \
                # "First stacking (torch.stack(trimmed_waveforms)): {}" \
                # "".format(waveform.data.shape, mono.shape, 
                #             len(trimmed_waveforms),(trimmed_waveforms[0].shape, 
                #                                     trimmed_waveforms[1].shape),
                #                                     source_tensor.shape))

        # 5. After the 'for' loop ends a global loop comes into effect,
        # ... whose purpose is to level down all the samples
        if self.dynamic:
            global_min_length = min([w.shape[1] for w in all_simulated_waveforms])
        else:
            global_min_length=4800

        # 5.1 Compute the length of each waveform by using a conditioned list comprehension
        # 5.1.1 IF waveform length higher than global_min_length then chunk off the tail
        all_simulated_and_trimmed_waveforms = [w[:, :global_min_length] if w.shape[1] >= global_min_length 
        # 5.1.2 Else pad zeros starting from left to right
        else torch.nn.functional.pad(w, (0, global_min_length - w.shape[1])) for w in all_simulated_waveforms]
        
        # # Debug statement 3
        # print(all_simulated_waveforms)
        # print(all_simulated_and_trimmed_waveforms)
        # print("Initial len {} vs aftermath len{}".format(len(all_simulated_waveforms), len(all_simulated_and_trimmed_waveforms)))
        
        stacked_tensor = torch.stack(all_simulated_and_trimmed_waveforms) # Shape: [n_sources, n_mics, samples]
        mixed_tensor = torch.sum(stacked_tensor, dim=0)       # Shape: [n_mics, samples]      
        mixed_tensor = mixed_tensor - mixed_tensor.mean(dim=1, keepdim=True)
        scale = mixed_tensor.pow(2).mean().sqrt().clamp_min(1e-8)
        mixed_tensor=mixed_tensor/scale
        
        # 6 Redefining the torch shape structure to hold 7 elements, 
        # ... first 6 of which are representing X,Y pair coordinates for sources
        if self.n_sources[1]>1:
            # 6.1 Sort the sources by distance to the microphones (origin 0,0)
            source_locations.sort(key=lambda loc: loc[0]**2 + loc[1]**2)
            target_locations = torch.zeros(7, dtype=torch.float32) # 6.2 Creates the dummy tensor with the reuqired size
            # 6.3 Fill the fixed tensor with the active, sorted coordinates
            for i, (x, y) in enumerate(source_locations):
                target_locations[i * 2] = x
                target_locations[i * 2 + 1] = y
            # 6.4. Store the actual number of sources in the very last index
            target_locations[6] = n_sources
        
        else:
            target_locations = torch.tensor(source_locations).squeeze()
        
        # Informative staement
        # print("Second stacking:",stacked_tensor.shape)
        # print("Third stacking:",mixed_tensor.shape)
        # print("Source tensor shape {} Stacked tensor shape {} Source locations {}".format(source_tensor.shape, stacked_tensor.shape, source_locations))
        
        if self.dynamic:
            return source_tensor, stacked_tensor, source_locations, delays
        else:
             return mixed_tensor, target_locations
  
    def _retrieve_and_assemble_data_paths(self, info:bool=False):
        # 0. Declaring path containers;
        _p_dataset:list = []
        dataset_roots:dict = [r'utils\assignment_2_dataset\train',
                      r'utils\assignment_2_dataset\validation',
                      r'utils\assignment_2_dataset\test']
        
        # 1. Determining the number of data sets we're interested in
        # ... commonly stays 3: train, test, val;
        for set in range(len(dataset_roots)):
            # 1.1. Retriveing the .mp3 file names
            names = os.listdir(self.dataset_root_paths[set])
            _p_dataset.append([os.path.join(self.dataset_root_paths[set],item) for item in names])

        if info:
            print(f"Container holding pahts to .mp3 files holds the following content {_p_dataset}")

        return _p_dataset
