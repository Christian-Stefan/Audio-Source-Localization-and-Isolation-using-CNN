from utils.submission_imports import *
from utils.submission_helpers import *
from utils.submission_models import *


if __name__ == "__main__":
    argument_dsahboard = argparse.ArgumentParser()
    argument_dsahboard.add_argument("--load", action="store_true", help="Load saved weights/results and generate figures without retraining.")
    argument_dsahboard.add_argument("--sample_bird_path", default=r'utils\assignment_2_dataset\train\bird1.mp3',type=str,help="Path leading to the bird audio file")
    argument_dsahboard.add_argument("--loss_graphs_paths", default=[r"utils\results\The loss graph entailing train and validation losses - d.png", r"utils\results\The loss graph entailing the average validation expressed in terms of CM - d.png",r"utils\results\The loss graph entailing the model performance evolution.png"],type=list,help="Paths leading to the graphs portraying the performance of the first model")
    argument_dsahboard.add_argument("--def_arch", default=[[4, 32, 64, 64],[1, 2, 2], [7, 5, 5]],type=list,help="List encompassing the default hyperparameters for DefaultCNN",)
    argument_dsahboard.add_argument("--model_weights_path", default=r"utils\results\model_Weights.pth", type=str, help="Stored paths wherefrom model weights are being loaded")
    argument_dsahboard.add_argument("--BATCH_SIZE", default=64,type=int)
    argument_dsahboard.add_argument("--SPACE_COORD", default=[(-200.0, 200.0),(25.0, 200.0)],type=list,help="Cartesian space coordinates stored in immutable structures")
    argument_dsahboard.add_argument("--DATASET_ROOTS", default=[r'utils\assignment_2_dataset\train', r'utils\assignment_2_dataset\validation', r'utils\assignment_2_dataset\test'],type=list,help="List holding train, validaton and test paths")
    argument_dsahboard.add_argument("--STEPS_PER_EPOCH", default=40,type=int,help="Number of gradient steps executed per epoch")
    argument_dsahboard.add_argument("--loss_graph_paths_3_sources", default=[r"utils\results\The CM loss graph entailing train and validation losses for an acustic environment with 3 sources.png",r"utils\results\The loss graph entailing train and validation  losses for an acustic environment with 3 sources.png"], type=list, help="List holding train, validaiton and test paths for N sources")
    argument_dsahboard.add_argument("--complex_arch", default=[[4, 32, 64, 128, 256], [128, 7, 5, 5, 3], [4, 4, 2, 2, 1]],type=list,help="List encompassing the optimal hyperparameters for ComplexCNN")
    args = argument_dsahboard.parse_args()

    if args.load:
        print("Running exercise_3.py in --load mode.")
        print("=================================== Loading results for 2.3.c) ===================================")
            # 1. Baseline initialization enforcing x,y global bounds (-200, 200);(-25,200) and some other variables (e.g., source path)
        input:str = args.sample_bird_path
        waveform, sample_rate = torchaudio.load(input)
        # 1.2. Averaging out the signal's channels 
        mono_waveform = torch.mean(waveform, dim=0) # Apply transformation
        mic_locs:list=[(-45, 0), (-15,0), (15, 0),(45,0)]
        fixed_locations:list = [(0, 50), (-100, 50), (0, 150), (100, 150)] # List of fixed locations
        lb_x, up_x = -200, 200
        lb_y, up_y = 25, 200
        fixed_locations_time_delays:list=[]
        # # 2. Parsing the list holding fixed coordinates and saving their time delays integers;
        # # ... Only one song was retained for the sake of simplicity
        # 2.1 Building the 2D space
        x = np.linspace(lb_x, up_x, 200)
        y = np.linspace(lb_y, up_y, 200)
        xv, yv = np.meshgrid(x, y) # Return a tuple (`xv` and `yv`) of coordinates matrices from coordinates vectors
        z = np.zeros_like(xv) # Create in advance the required z whose dimension matches the 2D tuple of coordinates; could be either of the two xv and yv

        # # 3. Parsing over the space coordinates and compare against delays calculated based on fix locations
        for loc in fixed_locations:
            _, delays = audio_simulator(mono_waveform, loc, mic_locs)
            # 3.1 Calculate difference relative to the first microphone
            relative_sig = [delays[1]-delays[0], delays[2]-delays[0], delays[3]-delays[0]]
            fixed_locations_time_delays.append(relative_sig)

        for i in range(xv.shape[0]):
            for j in range(xv.shape[1]):
                
                pixel_x = xv[i, j]
                pixel_y = yv[i, j]
                
                # Run your simulator for this one pixel to get its 4-mic delay list
                _, delays = audio_simulator(mono_waveform, (pixel_x, pixel_y), mic_locs)
                
                # 5. Check if this pixel perfectly matches any of our 4 target signatures
                # 5.1 Calculate pixel difference relative to the first microphone
                pixel_relative_sig = [delays[1]-delays[0], delays[2]-delays[0], delays[3]-delays[0]]
                # 5.2  compare the relative signatures!
                if pixel_relative_sig == fixed_locations_time_delays[0]:
                    z[i, j] = 1
                elif pixel_relative_sig == fixed_locations_time_delays[1]:
                    z[i, j] = 2
                elif pixel_relative_sig == fixed_locations_time_delays[2]:
                    z[i, j] = 3
                elif pixel_relative_sig == fixed_locations_time_delays[3]:
                    z[i, j] = 4
                # If it doesn't match any, it stays 0 (background)

        # # 4. Plot
        PLT.contourf(xv, yv, z, levels=4, cmap='Set1')
        PLT.colorbar(label='Target Area Match')

        # # 4.1 Plot the 4 actual test points as black stars so you can see them on the map
        test_x = [loc[0] for loc in fixed_locations]
        test_y = [loc[1] for loc in fixed_locations]
        PLT.scatter(test_x, test_y, color='black', marker='*', s=150, label='Sources (test points)')

        # # 4.2 Plot the 4 microphones as blue triangles
        mic_x = [loc[0] for loc in mic_locs]
        mic_y = [loc[1] for loc in mic_locs]
        PLT.scatter(mic_x, mic_y, color='blue', marker='^', s=100, label='Microphones')
        PLT.xlabel('Cartesian coordinates on X')
        PLT.ylabel('Cartesian coordinates on Y')

        PLT.legend()
        PLT.savefig(r'utils\results\Figure 1 - Spatial Quantization of Relative Time Delays.png', format='png')
        PLT.show()

        print("=================================== Loading results for 2.3.d) ===================================")
        img1 = mpimg.imread(args.loss_graphs_paths[0])
        img2 = mpimg.imread(args.loss_graphs_paths[1])

        # Create a figure frame with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the first image in the first slot
        axes[0].imshow(img1)
        axes[0].axis('off')  # Hide axes lines
        axes[0].set_title("MSE")

        # Display the second image in the second slot
        axes[1].imshow(img2)
        axes[1].axis('off')  # Hide axes lines
        axes[1].set_title("CM")

        # Show the frame containing both images
        PLT.tight_layout()
        PLT.show()

        print("=================================== Loading results for 2.3.e) ===================================")
        model = DefaultCNN(3, args.def_arch[0], args.def_arch[1], args.def_arch[2], sum=False)
        receptive_field = summary(model,(4, 4800), receptive_field=True)

        print("=================================== Loading results for 2.3.f) ===================================")
        img1 = mpimg.imread(args.loss_graphs_paths[2])
        PLT.imshow(img1)
        PLT.axis('off')
        # Show the frame containing both images
        PLT.tight_layout()
        PLT.show()

        # Calculate the exact number of samples needed
        total_train_samples = args.BATCH_SIZE * args.STEPS_PER_EPOCH
        # For validation, you usually want fewer steps (e.g., 10 batches) to save time
        total_val_samples = args.BATCH_SIZE * args.STEPS_PER_EPOCH

        train_dataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='train',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_train_samples)
        val_dataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='val',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_val_samples)
        test_dataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='test',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_val_samples)
        trainLoader = DataLoader(train_dataset,batch_size=args.BATCH_SIZE, shuffle=True)
        valLoader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
        testLoader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
        mins = torch.tensor([[args.SPACE_COORD[0][0], args.SPACE_COORD[1][0]]], dtype=torch.float32)
        maxs = torch.tensor([[args.SPACE_COORD[0][1], args.SPACE_COORD[1][1]]], dtype=torch.float32)

        optim = Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
        loss = MSELoss()
        # 2.1 (Optional best practice: add weights_only=True for security)
        weights = torch.load(args.model_weights_path, weights_only=True) 
        model.load_state_dict(weights)
        # 2.2  Set to evaluation mode (only needs to be called once)
        model.eval() 
        test_loss, test_loss_in_cm = 0.0, 0.0
        # 3. Evaluate exactly once 
        with torch.no_grad():
            for waveform, source_loc in testLoader: 
                # 3.1 Calculate standard normalized loss
                norm_source_loc = normalize_targets(source_loc, mins, maxs)
                _output_val = model(waveform)
                _loss_val = loss(_output_val, norm_source_loc)
                # 3.2 Calculate real-world Centimeter loss
                preds_val_cm = denormalize_predictions(_output_val, mins, maxs)
                _loss_val_cm = torch.sqrt(loss(preds_val_cm, source_loc))
                # 3.3 Accumulate batch metrics
                test_loss += _loss_val.item()
                test_loss_in_cm += _loss_val_cm.item()

        # 4. Average the final metrics
        avg_test_loss = test_loss / len(testLoader)
        avg_test_loss_cm = test_loss_in_cm / len(testLoader)

        print(f"Final Test MSE Loss: {avg_test_loss:.4f}")

        print("=================================== Loading results for 2.3.g) ===================================")
        img1 = mpimg.imread(args.loss_graph_paths_3_sources[0])
        img2 = mpimg.imread(args.loss_graph_paths_3_sources[1])

        # Create a figure frame with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the first image in the first slot
        axes[0].imshow(img1)
        axes[0].axis('off')  # Hide axes lines
        axes[0].set_title("CM")

        # Display the second image in the second slot
        axes[1].imshow(img2)
        axes[1].axis('off')  # Hide axes lines
        axes[1].set_title("MSE")

        # Show the frame containing both images
        PLT.tight_layout()
        PLT.show()

        print("=================================== Loading results for 2.3.i) ===================================")
        # 1.Load the model
        arch, kernel_sizes, strides, mic_locs = [4, 32, 64, 128, 256], [128, 7, 5, 5, 3], [4, 4, 2, 2, 1], [(-45, 0), (-15,0), (15, 0),(45,0)]
        model = NSourcesCNN(4, arch, strides, kernel_sizes, sum=False)
        for n_source in range(3):

            # 2.1 Create a new data retrieval that will hold between 1-3 sources;
            train_1_sourcesdataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=n_source+1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='train',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_train_samples)
            val_1_sourcesdataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=n_source+1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='val',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_val_samples)
            test_1_sourcesdataset=CustomAudioDecoderData(t=100,n_sources_min=1, n_sources_max=n_source+1, x_loc_min=-200, x_loc_max=200,y_loc_max=200,y_loc_min=25, split='test',mic_locs=mic_locs, data_folder=args.DATASET_ROOTS, transforms=True, dynamic=False, epoch_len=total_val_samples)
            # 2.2 Consequently, create new loaders
            n1sources_trainLoader = DataLoader(train_1_sourcesdataset,batch_size=args.BATCH_SIZE, shuffle=True)
            n1sources_valLoader = DataLoader(val_1_sourcesdataset, batch_size=args.BATCH_SIZE, shuffle=False)
            n1sources_testLoader = DataLoader(test_1_sourcesdataset, batch_size=args.BATCH_SIZE, shuffle=False)

            # 2.1 (Optional best practice: add weights_only=True for security)
            weights = torch.load(r'utils\results\N_sources_model_weights.pth', weights_only=True) 
            model.load_state_dict(weights)

            X_MIN, X_MAX = -200.0, 200.0
            Y_MIN, Y_MAX = 25.0, 200.0
            mins_2 = torch.tensor([[X_MIN, Y_MIN]], dtype=torch.float32) # Creates tensor of shape (1,2) holding lower bounds coordinates
            maxs_2 = torch.tensor([[X_MAX, Y_MAX]], dtype=torch.float32) # -----------------=====------- holding upper bounds cordinates
            mins_6 = mins_2.repeat(1, 3) # Repeat [X, Y] three times to match the [X1, Y1, X2, Y2, X3, Y3] shape
            maxs_6 = maxs_2.repeat(1, 3)

            # 2.2  Set to evaluation mode (only needs to be called once)
            epochs, test_1batch, test_1batch_cm=15, [],[]
            # 3. Evaluate exactly once 
            # 3. Evaluate exactly once 
            test_loss, test_loss_in_cm = 0.0, 0.0    
            with torch.no_grad():
                model.eval()
                for waveform, raw_targets in n1sources_testLoader:
                    
                    # --- NEW PADDING LOGIC ---
                    batch_size_curr = raw_targets.size(0)
                    target_len = raw_targets.size(1)
                    
                    if target_len < 7:
                        # 1. Figure out how many sources this is (e.g., length 2 = 1 source)
                        actual_n_sources = target_len // 2
                        count_tensor = torch.full((batch_size_curr, 1), actual_n_sources, dtype=torch.float32, device=raw_targets.device)
                        
                        # 2. Create the 6-coordinate placeholder filled with zeros
                        padded_coords = torch.zeros((batch_size_curr, 6), dtype=torch.float32, device=raw_targets.device)
                        
                        # 3. Copy the valid coordinates into the placeholder
                        padded_coords[:, :target_len] = raw_targets
                        
                        # 4. Stitch it together to recreate the 7-element tensor
                        packed_targets = torch.cat([padded_coords, count_tensor], dim=1)
                    else:
                        packed_targets = raw_targets
                    # -------------------------

                    # 7.1 No. of sources and coordinates retrieval (Now perfectly safe!)
                    target_coords = packed_targets[:, :6]
                    num_sources = packed_targets[:, 6]
                    
                    _output_val = model(waveform)
                    pred_coords = _output_val[:, :6]
                    pred_count = _output_val[:, 6]
                    
                    # 7.2 Mask Creation
                    batch_size = num_sources.size(0)
                    idx = torch.arange(6, device=waveform.device).unsqueeze(0).expand(batch_size, 6)
                    active_lengths = (num_sources * 2).unsqueeze(1)
                    mask = (idx < active_lengths).float()
                    
                    # 7.3 Masked Val Coordinate Loss
                    norm_target_coords = normalize_targets(target_coords, mins_6, maxs_6)
                    masked_coord_loss = loss(pred_coords, norm_target_coords) * mask
                    loss_coords = torch.sum(masked_coord_loss) / torch.sum(mask).clamp_min(1e-8)
                    
                    # 7.4 Val Count Loss
                    loss_count = torch.mean((pred_count - num_sources) ** 2)
                    _test_val = loss_coords + (0.05 * loss_count)
                    
                    # 7.5 Val CM Error
                    preds_cm = denormalize_predictions(pred_coords, mins_6, maxs_6)
                    masked_cm_loss = loss(preds_cm, target_coords) * mask
                    _test_val_cm = torch.sqrt(torch.sum(masked_cm_loss) / torch.sum(mask).clamp_min(1e-8))
                    
                    test_loss += _test_val.item()
                    test_loss_in_cm += _test_val_cm.item()

            # 4. Average the final metrics (Fixed the variable name here to n1sources_testLoader!)
            avg_test_loss = test_loss / len(n1sources_testLoader)
            test_1batch.append(avg_test_loss)
            
            avg_test_loss_cm = test_loss_in_cm / len(n1sources_testLoader)
            test_1batch_cm.append(avg_test_loss_cm)

            print(f"Final Test MSE Loss with {n_source+1} source: {avg_test_loss:.4f}")
            print(f"Final Test Error in CM with {n_source+1} source: {avg_test_loss_cm:.2f} cm")

        print("=================================== Loading results for 2.3.j) ===================================")
        # 0.Load the model
        arch, kernel_sizes, strides, mic_locs = [4, 32, 64, 128, 256], [128, 7, 5, 5, 3], [4, 4, 2, 2, 1], [(-45, 0), (-15,0), (15, 0),(45,0)]
        model = NSourcesCNN(4, arch, strides, kernel_sizes, sum=False) 

        # 2.1 (Optional best practice: add weights_only=True for security)
        weights = torch.load(r'utils\results\N_sources_model_weights.pth', weights_only=True) 
        model.load_state_dict(weights)


        # 1. Set up the experiment
        test_file_1:str = r"utils\assignment_2_dataset\test\XC1040774.mp3" 
        test_file_2:str = r"utils\assignment_2_dataset\test\XC1043067.mp3" 


        # 2 Decode the full files 
        # ... (e.g., first 10 seconds to keep plotting manageable)
        # 2.1 Decode the file
        duration:float = 10.0 
        decoder1 = AudioDecoder(test_file_1)
        decoder2 = AudioDecoder(test_file_2)
        wave1 = decoder1.get_samples_played_in_range(0, duration).data.mean(dim=0)
        wave2 = decoder2.get_samples_played_in_range(0, duration).data.mean(dim=0)
        # 2.2 Simulate the delays for the specific coordinates
        # ... Simulate the delays for the specific coordinates (These return Python lists)
        sim_wave1_list, _ = audio_simulator(wave1, source_loc=(0.0, 50.0), mic_locs=mic_locs)
        sim_wave2_list, _ = audio_simulator(wave2, source_loc=(100.0, 150.0), mic_locs=mic_locs)
        # 2.2.1 Helper function to trim and stack the list of 4 microphones into a 2D tensor
        def prepare_mic_tensor(wave_list):
            min_l = min([w.shape[0] for w in wave_list])
            return torch.stack([w[:min_l] for w in wave_list])
        sim_wave1 = prepare_mic_tensor(sim_wave1_list) # Convert lists to 2D tensors -> Shape: [4, samples]
        sim_wave2 = prepare_mic_tensor(sim_wave2_list) # Convert lists to 2D tensors -> Shape: [4, samples]
        # 2.3 Mix them together
        min_len = min(sim_wave1.shape[1], sim_wave2.shape[1])
        mixed_wave = sim_wave1[:, :min_len] + sim_wave2[:, :min_len]
        # 3. THE SLIDING WINDOW INFERENCE ---
        window_size = 4800 # 0.1 seconds at 48kHz
        num_windows = min_len // window_size
        errors_b1, errors_b2 = [], []
        vols_b1, vols_b2 = [], []
        time_axis = []

        model.eval()
        with torch.no_grad():
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                
                # 2a. Process the chunk exactly like the DataLoader does
                chunk = mixed_wave[:, start:end]
                chunk = chunk - chunk.mean(dim=1, keepdim=True)
                scale = chunk.pow(2).mean().sqrt().clamp_min(1e-8)
                chunk = chunk / scale
                chunk = chunk.unsqueeze(0) # Add batch dimension -> [1, 4, 4800]
                
                # 2b. Make Prediction
                output = model(chunk)
                pred_coords = output[:, :6]
                preds_cm = denormalize_predictions(pred_coords, mins_6, maxs_6).squeeze()
                
                # 2c. Map Predictions to Targets (Because we sorted Left-to-Right!)
                # Target 1 is X=0, Target 2 is X=100. 
                # Therefore, the network's first output is Bird 1, second is Bird 2.
                pred_b1 = preds_cm[0:2] 
                pred_b2 = preds_cm[2:4]
                
                # 2d. Calculate CM Error
                err1 = torch.norm(pred_b1 - torch.tensor([0.0, 50.0])).item()
                err2 = torch.norm(pred_b2 - torch.tensor([100.0, 150.0])).item()
                errors_b1.append(err1)
                errors_b2.append(err2)
                
                # 2e. Calculate True Original Volume (RMS Energy) for this 0.1s window
                vol1 = wave1[start:end].pow(2).mean().sqrt().item()
                vol2 = wave2[start:end].pow(2).mean().sqrt().item()
                vols_b1.append(vol1)
                vols_b2.append(vol2)
                
                time_axis.append(start / 48000) # Convert samples back to seconds

        # --- 3. PLOTTING THE RESULTS ---
        fig, (ax1, ax2) = PLT.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Error
        ax1.plot(time_axis, errors_b1, label='Bird 1 Error (0, 50)', color='blue', alpha=0.7)
        ax1.plot(time_axis, errors_b2, label='Bird 2 Error (100, 150)', color='orange', alpha=0.7)
        ax1.set_ylabel('Localization Error (cm)')
        ax1.set_title('CNN Estimation Error Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot Volume
        ax2.plot(time_axis, vols_b1, label='Bird 1 True Volume', color='blue', alpha=0.7)
        ax2.plot(time_axis, vols_b2, label='Bird 2 True Volume', color='orange', alpha=0.7)
        ax2.set_ylabel('RMS Volume')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Original Unmixed Sound Volume Over Time')
        ax2.legend()
        ax2.grid(True)

        PLT.tight_layout()
        PLT.show()