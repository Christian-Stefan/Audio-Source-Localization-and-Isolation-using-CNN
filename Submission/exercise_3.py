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