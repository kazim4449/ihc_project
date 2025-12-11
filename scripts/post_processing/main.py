# You may run this using python3 -m scripts.post_processing.main --version=1 --continue_training --tl_version=1 from ihc_project folder. 
# This can evaluate the best model of v001 cross validation set and give metrics associated to the model 
# which is then saved to model_epoch_{epoch_num}_results folder. Further it can create the sample 
# predictons for all the images in the validation sets, along with the per image matric and confusion matrix. 
# Moreover it can plot the training curves from the training.json file, by specifying the best epoch. 
# Please note that the best epochs are manually entered in line 267. So if there is a change, update it manually 
# if we add argument continue_learning it will use the IHC data to continue training using transfer learning. 
# the argument use_model_before_transfer_learning can be used to obtain the prediction results of the H&E model using IHC test data 
# if we set the argument use_IHC_train_data, the prediction results of using H&E model on IHC train data is produced. 
# now we have cross validation on continue_training part. We should also give the trasnfer learning cross validation set 
# for this we are adding one more argument. tl_version (default is 001). The values can vary from 001 to 010. 

import os



import glob
import os
import argparse
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.ticker import MaxNLocator
class Testing_pipeline:
    def __init__(self,version, epoch, continue_training = False, tl_version = '001', use_model_before_transfer_learning = False, use_IHC_train_data = False):
        self.continue_training = continue_training
        self.before_transfer_learning = use_model_before_transfer_learning
        self.tl_version = tl_version
        # use IHC train data for testing the HnE model performance on IHC train
        self.use_IHC_train_data = use_IHC_train_data
        self.PROJECT_ROOT = os.path.abspath(os.getcwd())
        self.version = version
        if continue_training:
            cont_suffix = '_cont1/training_'+self.tl_version
            self.model_base_path = "models/resnet152_"+version+ cont_suffix
            self.results_dir = self.model_base_path+"/epochs" 
            self.model_path = "models/resnet152_"+version + "/epochs" # model from which the training is continued
            #self.save_path = model_base_path+ f"/model_epoch_{self.epoch_str}_results"
            self.history_file_path = self.model_base_path+"/training_history.json"
            self.epoch  = epoch[1]
            self.initial_epoch = epoch[0] +1
        else: 
            self.results_dir = "models/resnet152_"+version+"/epochs" 
            self.model_base_path = "models/resnet152_"+version
           # self.save_path = "models/resnet152_"+ version+ f"/model_epoch_{self.epoch_str}_results"
            self.version_directory =  "models/resnet152_"+ self.version
            self.history_file_path = os.path.join(self.PROJECT_ROOT,self.version_directory,"training_history.json")
            self.epoch = epoch
            self.initial_epoch= 1
        self.epoch_str = f"{self.epoch:02d}"  # '07'
        self.save_path = self.model_base_path+ f"/model_epoch_{self.epoch_str}_results"
        self.pred_dir = os.path.join(self.save_path, "sample_predictions")
        self.eval_dir = os.path.join(self.save_path, "evaluation")
        print('Epoch string is ' + self.epoch_str)
        # Create the folder (and parent folders if needed)
        if use_model_before_transfer_learning == False:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.pred_dir, exist_ok=True)
            os.makedirs(self.eval_dir, exist_ok=True)

    
    def load_model(self):
        # load model 
        import keras
        if self.continue_training:
            # to check if HnE trained model is enough for IHC data as well or not, we load for eg: model 19 for version 1 and test on IHC data 
            if self.before_transfer_learning:
                model_path = "models/resnet152_"+self.version + "/epochs"
                epoch_str = f"{(self.initial_epoch-1):02d}"
                print('Epoch string is ' + epoch_str)
                if self.use_IHC_train_data:
                    self.save_path = self.model_base_path+ f"/HnE_model_epoch_{epoch_str}_results/on_train_data"
                else:
                    self.save_path = self.model_base_path+ f"/HnE_model_epoch_{epoch_str}_results/on_test_data"
                self.pred_dir = os.path.join(self.save_path, "sample_predictions")
                self.eval_dir = os.path.join(self.save_path, "evaluation")
                # Create the folder (and parent folders if needed)
                os.makedirs(self.save_path, exist_ok=True)
                os.makedirs(self.pred_dir, exist_ok=True)
                os.makedirs(self.eval_dir, exist_ok=True)
            else:
                model_path = self.results_dir
                epoch_str = f"{(self.epoch):02d}"  # '07'
        else:
            model_path = self.results_dir
            epoch_str = self.epoch_str
        files = glob.glob(f"{model_path}/resnet152_epoch{epoch_str}*.keras")
        print(files)
        print(model_path)
        #print(self.results_dir)
        #print(f"{self.results_dir}/resnet152_epoch{self.epoch_str}*.keras")
        if files.__len__()==1:
            model =  keras.models.load_model(files[0],compile=False)
        else:
            raise ValueError("Expected exactly one file, but got {}.".format(len(files)))
        return model

    def evaluate_model_and_make_visualisations(self):
        from . import images_to_tensors as i2t
        from . import full_eval_metrics as em_f
        from . import predict_visualize as pdv
        # load model 
        model = self.load_model()
         # convert images to tensors 
        image_to_tensor =  i2t.ImagetoTensor(self.version,self.continue_training, self.tl_version, self.use_IHC_train_data)
        X_val, y_val = image_to_tensor.preprocess_data(*image_to_tensor.load_images_and_masks())
        n_samples = X_val.shape[0]
        # visualize predictions and calculate metrics for all test images
        pdv.visualize_predictions(model, X_val, y_val, n_samples, save_path=self.pred_dir)
        # evaluate by finding the general metrics of the model 
        em_f.evaluate_metrics(model, X_val, y_val, self.eval_dir)

    def plot_line_curves_from_history(self):
        # -----------------------------------------------------------------
        # line curves of loss, IOU_score and F1_score from training_history
        # -----------------------------------------------------------------

        # plotting every parameters inside the training_history.json vs epoch in a line diagram

        epoch_number  = int(self.epoch)
        if self.continue_training:
            real_best_epoch = epoch_number-self.initial_epoch+1

        df = pd.read_json(self.history_file_path)
        #print(df)
        ''' To plot from initial epoch to final epoch where initial epoch = epoch from first training +1
        #---------------
        # Loss per epoch
        #---------------

        ax =  df['loss'].set_axis(range(self.initial_epoch, self.initial_epoch+len(df))).plot(label = 'Training_loss')
        
        # plot another column in the same axes
        df['val_loss'].set_axis(range(self.initial_epoch, self.initial_epoch+len(df))).plot(ax = ax, label= 'Validation_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.title('Loss per epoch')
        # Add vertical line at x=5
        plt.axvline(x=real_best_epoch, color='red', linestyle='--', label='x = '+str(epoch_number))
        loss_file = os.path.join(self.save_path, 'loss.png') 
        # Force integer ticks on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(loss_file)
        print(f"Loss saved to loss.png")
        plt.close()

        #-------------------
        #IOU_Score per Epoch
        #-------------------
        ax2 =  df['iou_score'].set_axis(range(self.initial_epoch,self.initial_epoch+ len(df))).plot(label = 'Training_IOU_Score')

        # plot another column in the same axes
        df['val_iou_score'].set_axis(range(self.initial_epoch, self.initial_epoch+len(df))).plot(ax = ax2, label= 'Validation_IOU_Score')
        plt.xlabel('Epoch')
        plt.ylabel('IOU_Score')
        plt.legend()
        plt.title('IOU_Score per epoch')
        plt.axvline(x=epoch_number, color='red', linestyle='--', label='x = '+str(epoch_number))
        IOU_file = os.path.join(self.save_path, 'IOU_score.png') 
        # Force integer ticks on the x-axis
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(IOU_file)
        print(f"IOU Score saved to IOU_score.png")
        plt.close()

        #------------------
        #F1_Score per epoch
        #------------------
        ax3 =  df['f1-score'].set_axis(range(self.initial_epoch,self.initial_epoch+ len(df))).plot(label = 'Training_F1_Score')

        # plot another column in the same axes
        df['val_f1-score'].set_axis(range(self.initial_epoch, self.initial_epoch+len(df))).plot(ax = ax3, label= 'Validation_F1_Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1_Score')
        
        plt.title('F1_Score per epoch')
        plt.axvline(x=epoch_number, color='red', linestyle='--', label='x = '+str(epoch_number))
        plt.legend()
        F1_file = os.path.join(self.save_path, 'F1_score.png') 
        # Force integer ticks on the x-axis
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(F1_file)
        print(f"F1 score saved to F1_score.png")
        plt.close()

         '''
        
        # To plot the loss, F1 and IOU score per epoch in the case of the initial epoch counted from 1 and nopt last epoch from H&E training+1 
        #---------------
        # Loss per epoch
        #---------------

        ax =  df['loss'].set_axis(range(1, 1+len(df))).plot(label = 'Training_loss')
        
        # plot another column in the same axes
        df['val_loss'].set_axis(range(1, 1+len(df))).plot(ax = ax, label= 'Validation_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.title('Loss per epoch')
        # Add vertical line at x=5
        plt.axvline(x=real_best_epoch, color='red', linestyle='--', label='x = '+str(real_best_epoch))
        loss_file = os.path.join(self.save_path, 'loss.png') 
        # Force integer ticks on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(loss_file)
        print(f"Loss saved to loss.png")
        plt.close()

        #-------------------
        #IOU_Score per Epoch
        #-------------------
        ax2 =  df['iou_score'].set_axis(range(1,1+ len(df))).plot(label = 'Training_IOU_Score')

        # plot another column in the same axes
        df['val_iou_score'].set_axis(range(1, 1+len(df))).plot(ax = ax2, label= 'Validation_IOU_Score')
        plt.xlabel('Epoch')
        plt.ylabel('IOU_Score')
        plt.legend()
        plt.title('IOU_Score per epoch')
        plt.axvline(x=real_best_epoch, color='red', linestyle='--', label='x = '+str(real_best_epoch))
        IOU_file = os.path.join(self.save_path, 'IOU_score.png') 
        # Force integer ticks on the x-axis
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(IOU_file)
        print(f"IOU Score saved to IOU_score.png")
        plt.close()

        #------------------
        #F1_Score per epoch
        #------------------
        ax3 =  df['f1-score'].set_axis(range(1,1+ len(df))).plot(label = 'Training_F1_Score')

        # plot another column in the same axes
        df['val_f1-score'].set_axis(range(1, 1+len(df))).plot(ax = ax3, label= 'Validation_F1_Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1_Score')
        
        plt.title('F1_Score per epoch')
        plt.axvline(x=real_best_epoch, color='red', linestyle='--', label='x = '+str(real_best_epoch ))
        plt.legend()
        F1_file = os.path.join(self.save_path, 'F1_score.png') 
        # Force integer ticks on the x-axis
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(F1_file)
        print(f"F1 score saved to F1_score.png")
        plt.close()

    def plot_important_part_from_history(self):
        
        epoch_number  = int(self.epoch)
        #print(history_file_path)
        df = pd.read_json(self.history_file_path)
        # 1. Find the position (not index) of the minimum loss
        min_pos = df['val_loss'].values.argmin()  # Gives position like 0, 1, 2...
        print(df)
        print(min_pos)
        # 2. Calculate safe start and end positions
        start = max(min_pos - 8, 0)
        end = min(min_pos + 8, len(df) - 1)
        print(start)
        print(end)
        # 3. Slice the DataFrame using iloc (position-based slicing)
        window = df['loss'].iloc[start:end + 1]

        # 4. Reset the index for clean x-axis (1 to N)
        window.index = range(start+1, end + 2)

        # 5. Plot
        ax = window.plot(label='Loss ±8 around min', logy = True)
        # plot another column in the same axes
        df['val_loss'].iloc[start:end + 1].set_axis(window.index).plot(ax = ax, label= 'Validation_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        plt.axvline(x=epoch_number, color='red', linestyle='--', label='x = '+str(epoch_number))
        loss_file = os.path.join(self.save_path, 'scaled_loss.png') 
        ax.legend()
        plt.savefig(loss_file)
        print(f"Scaled_Loss saved to scaled_loss.png")
        plt.close()

    def plot_scaled_training_curves(self):
        epoch_number  = int(self.epoch)
        #print(history_file_path)
        df = pd.read_json(self.history_file_path)
        #---------------
        # Loss per epoch
        #---------------  

        def exp_smooth(xs, alpha=0.2):
            sm = []
            s = xs[0]
            for x in xs:
                s = alpha * x + (1 - alpha) * s
                sm.append(s)
            return sm
         
        plt.plot(range(1,len(df)+1),exp_smooth(df['loss'] ,alpha=0.2), label = 'Training_loss')     
        plt.plot(range(1,len(df)+1),exp_smooth(df['val_loss'], alpha=0.2), label = 'Validation_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
             

        plt.axvline(x=epoch_number, color='red', linestyle='--', label='x = '+str(epoch_number))
        loss_file = os.path.join(self.save_path, 'scaled_loss_fig.png') 
        plt.legend()   
        plt.savefig(loss_file)
        print(f"Scaled_Loss saved to scaled_loss_fig.png")
        plt.close()

    def comparing_test_samples(self):
        # ---------------------------------------------------------------------------------------------------------
        # Plotting the number of correct vs incorrect predictions for each class and in general for all test images
        # ---------------------------------------------------------------------------------------------------------

        # load the metrics.txt file for each images and get the values inside. 

        import pandas as pd
        import glob
        import file_utils

            # choose which epoch you want to examine
            # eg: 4


        # each of these files contain the per class IOU and weighted precision, recall and F1 score per test case
        metric_files = glob.glob( f'{self.save_path}/sample_predictions/sample_*_metrics.txt')
        # Sort numerically by the number after 'sample_'
        #files = sorted(metric_files, key=lambda x: int(re.search(r'sample_(\d+)_metrics', x).group(1)))
        files = file_utils.natural_sort(metric_files)
        print(files)
        W_Precision, W_Recall, W_F1, IOU_class0, IOU_class1, IOU_class2, IOU_class3 = [], [], [], [], [], [], []
        for i in range(files.__len__()):
            df2 = pd.read_csv(files[i], sep=":", header=None, names=["key", "value"])
            W_Precision.append(df2.loc[df2["key"] == "Weighted Precision", "value"].iloc[0])
            W_Recall.append(df2.loc[df2["key"] == "Weighted Recall", "value"].iloc[0])
            W_F1.append(df2.loc[df2["key"] == "Weighted F1", "value"].iloc[0])
            IOU_class0.append(df2.loc[df2["key"] == "Class 0", "value"].iloc[0])
            IOU_class1.append(df2.loc[df2["key"] == "Class 1", "value"].iloc[0])
            IOU_class2.append(df2.loc[df2["key"] == "Class 2", "value"].iloc[0])
            IOU_class3.append(df2.loc[df2["key"] == "Class 3", "value"].iloc[0])
            print(df2)

        import matplotlib.pyplot as plt
        import seaborn as sns


        label1 = [df2.key[0], df2.key[1], df2.key[2], 'IOU_Score '+df2.key[4], 'IOU_Score '+df2.key[5], 'IOU_Score '+df2.key[6], 'IOU_Score '+df2.key[7] ] 
        values = [W_Precision, W_Recall, W_F1, IOU_class0, IOU_class1, IOU_class2, IOU_class3]
        for i in range(values.__len__()):
            plt.plot(values[i])
            plt.title(label1[i]+" Over Test Images")
            plt.xlabel("Test Image")
            plt.ylabel(label1[i])
            plt.show()

            sns.boxplot(data=values[i])
            plt.title("Boxplot of "+ label1[i])
            plt.ylabel(label1[i])
            plt.show()

            plt.hist(values[i], bins=10, edgecolor='black')
            plt.title("Histogram of "+ label1[i])
            plt.xlabel(label1[i])
            plt.ylabel("Frequency")
            plt.show()

# ----------------------
# Entry Point
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test segmentation model with different cross validation sets")
    parser.add_argument("--version", type=int, default="2", help="Version for the intended crossvalidation set")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from a saved checkpoint")
    parser.add_argument("--use_model_before_transfer_learning", action="store_true", help="Use the HnE trainined model to test IHC test data")
    parser.add_argument("--use_IHC_train_data", action="store_true", help="Use the HnE trainined model to test IHC train data")
    parser.add_argument("--tl_version",type=int,default = "1", help = "The cross validation set used in transfer learning (IHC)")
    args = parser.parse_args()
    #version = 2 # enter values between 1 to 10 

    best_epochs =  [19, 13, 22, 11, 18, 19, 20, 11, 14, 20]
    best_epochs_v001_tl = [31,55,53,35,45,43,40,63,47,42]
    best_epochs_v007_tl =[32,36,59,30,64,43,44,49,41,37]
    best_epochs_v001_tl_data_8 =[58]
    best_epochs_v007_tl_data_8 = [29]
    if args.continue_training:
        #best_epochs = [31,32]
        if args.version == 1:
            epoch = [19,best_epochs_v001_tl_data_8[args.tl_version-1]] # [initial_epoch,best epoch] tuple where initial epoch-1 is the last epoch from which we started training
        elif args.version == 7:
            epoch = [20,best_epochs_v007_tl_data_8[args.tl_version-1]] 
        else:
            print('Expected version 1 or 7 but got {args.version}')
        print('Best epoch is'+ str(epoch[1]) + 'and initial epoch is'+  str(epoch[0]))
    else:
        epoch = best_epochs[args.version-1]
        print('Best epoch is {epoch}')
    formatted_version = 'v'+f"{args.version:03}"
    formatted_tl_version = f"{args.tl_version:03}"
    Test = Testing_pipeline(formatted_version,epoch,args.continue_training,formatted_tl_version,args.use_model_before_transfer_learning,args.use_IHC_train_data)
    Test.plot_line_curves_from_history() 
    Test.evaluate_model_and_make_visualisations()


