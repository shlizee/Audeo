import os
import json
import numpy as np
import torch
import glob
from Roll2MidiNet import Generator
from torch.autograd import Variable
torch.cuda.set_device(0)
cuda = torch.device("cuda")
print(torch.cuda.current_device())
Tensor = torch.cuda.FloatTensor
class Midi_Generation():
    def __init__(self, checkpoint, exp_dir, est_roll_folder, video_name):
        # model dir
        self.exp_dir = exp_dir
        # load model checkpoint
        self.checkpoint = torch.load(os.path.join(exp_dir,checkpoint))
        # the video name
        self.video_name = video_name
        # the Roll prediction folder
        self.est_roll_folder = est_roll_folder + video_name
        # Midi output dir
        self.infer_out_dir = '/home/neuralnet/MIDI_GAN/Roll2Midi_results/testing'

        self.min_key = 15
        self.max_key = 65
        self.frame = 50
        self.process_est_roll(self.est_roll_folder)

    def process_est_roll(self, est_roll_folder):
        self.data = []
        self.final_data = []
        self.est_roll_files = glob.glob(est_roll_folder + '/*.npz')
        self.est_roll_files.sort(key=lambda x: int(x.split('/')[7].split('.')[0].split('-')[0]))
        print("need to infer {0} files".format(len(est_roll_folder)))
        for i in range(len(self.est_roll_files)):
            with np.load(self.est_roll_files[i]) as data:
                est_roll = data['logit'][:,self.min_key:self.max_key+1]
                if est_roll.shape[0] != self.frame:
                    target = np.zeros((self.frame, self.max_key-self.min_key+1))
                    target[:est_roll.shape[0], :] = est_roll
                    est_roll = target
            self.data.append(torch.from_numpy(est_roll))
        for i in range(0,len(self.data), 2):
            if i + 1 < len(self.data):
                one_roll = self.data[i]
                two_roll = self.data[i+1]
                final_roll = torch.cat([one_roll, two_roll], dim=0)
                self.final_data.append(final_roll)

    def inference(self):
        input_shape = (1, self.max_key-self.min_key+1, 2*self.frame)
        model = Generator(input_shape).cuda()
        model.load_state_dict(self.checkpoint['state_dict_G'])
        test_results = []
        print('Inferencing MIDI......')
        for i, data in enumerate(self.final_data):
            roll = torch.unsqueeze(torch.unsqueeze(torch.sigmoid(data.T.float().cuda()), dim=0), dim=0)
            print("piece ", i)
            with torch.no_grad():
                model.eval()
                roll = roll.type(Tensor)
                roll_ = Variable(roll)
                gen_img = model(roll_)
                gen_img = gen_img >= 0.5

                numpy_pre_label = gen_img.cpu().detach().numpy().astype(np.int) # 1,1,88,100
                numpy_pre_label = np.transpose(numpy_pre_label.squeeze(), (1, 0))  # 100,88

                test_results.append(numpy_pre_label[:self.frame, :])
                test_results.append(numpy_pre_label[self.frame:, :])
        midi_out_dir = self.create_output_dir()
        for i in range(len(test_results)):
            print(self.est_roll_files[i])
            idx = self.est_roll_files[i].split("/")[-1].split(".")[0].split("-")
            idx1 = int(idx[0])
            idx2 = int(idx[1])
            print(idx1, idx2)
            np.savez(midi_out_dir+f'/{idx1}-{idx2}.npz', midi=test_results[i])

    def create_output_dir(self):
        midi_out_dir = os.path.join(self.infer_out_dir, self.video_name)
        os.makedirs(midi_out_dir, exist_ok=True)
        return midi_out_dir

if __name__ == "__main__":
    # example for generating the Midi output from training Roll predictions
    est_roll_folder = './estimate_Roll/testing/'
    exp_dir = os.path.join(os.path.abspath('./experiments'), 'exp_5')
    with open(os.path.join(exp_dir,'hyperparams.json'), 'r') as hpfile:
        hp = json.load(hpfile)
    print("the best loss:", hp['best_loss'])
    print("the best epoch:", hp['best_epoch'])

    checkpoints = 'checkpoint-{}.tar'.format(hp['best_epoch'])
    for i in range(1, 4):
        video_name = f'Bach Prelude and Fugue No.{i} B2'
        generator = Midi_Generation(checkpoints, exp_dir, est_roll_folder, video_name)
        generator.inference()

