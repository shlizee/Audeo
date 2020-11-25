import Video2RollNet
import os
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
transform = transforms.Compose([lambda x: x.resize((900,100)),
                               lambda x: np.reshape(x,(100,900,1)),
                               lambda x: np.transpose(x,[2,0,1]),
                               lambda x: x/255.])

# video images root dir, change to your path
img_root='./input_images'
# labels root dir, change to your path
label_root='./labels'
# midi ground truth root dir, change to your path
midi_root = './midi'
# Roll prediction output, change to your path
est_roll_root = './estimate_Roll'

# the range of Piano keys (maximum is 88), depending on your data
min_key = 15
max_key = 65

def load_data(img_folder, label_file,midi_folder):
    img_files = glob.glob(img_folder + '/*.jpg')
    img_files.sort(key=lambda x: int(x.split('/')[4].split('.')[0][5:]))
    labels = np.load(label_file, allow_pickle=True)
    # Midi info for every video is divided into multiple npz files
    # each npz contains 2 seconds (50 frames) Midi information
    # format: frame_{i}-frame_{i+50}.npz
    midi_files = glob.glob(midi_folder + '/*.npz')
    midi_files.sort(key=lambda x: int(x.split('/')[4].split('.')[0].split('-')[0]))
    intervals = []
    for file in midi_files:
        interval = file.split('/')[4].split('.')[0].split('-')
        start = int(interval[0])
        end = int(interval[1])
        intervals.append([start, end])
    data = []
    for i, file in enumerate(img_files):
        key = int(file.split('/')[4].split('.')[0][5:])
        label = np.where(labels[key] > 0, 1, 0)
        new_label = label[min_key:max_key + 1]
        if i >= 2 and i < len(img_files) - 2:
            file_list = [img_files[i - 2], img_files[i - 1], file, img_files[i + 1], img_files[i + 2]]
        elif i < 2:
            file_list = [file, file, file,  img_files[i + 1], img_files[i + 2]]
        else:
            file_list = [img_files[i - 2], img_files[i - 1], file, file, file]
        data.append((file_list, new_label))
    return intervals, data

# infer 2 seconds every time
def inference(net, intervals, data, est_roll_folder):
    net.eval()
    i = 0
    for interval in intervals:
        start, end = interval
        print("infer interval {0} - {1}".format(start, end))
        save_est_roll = []
        save_est_logit = []
        infer_data = data[i:i+50]
        for frame in infer_data:
            file_list, label = frame
            torch_input_img, torch_label = torch_preprocess(file_list, label)
            logits = net(torch.unsqueeze(torch_input_img,dim=0))
            pred_label = torch.sigmoid(logits) >= 0.4
            numpy_pre_label = pred_label.cpu().detach().numpy().astype(np.int)
            numpy_logit = logits.cpu().detach().numpy()
            save_est_roll.append(numpy_pre_label)
            save_est_logit.append(numpy_logit)
        # Roll prediction
        target = np.zeros((50, 88))
        target[:, min_key:max_key+1] = np.asarray(save_est_roll).squeeze()
        save_est_roll = target
        # Logit
        target_ = np.zeros((50, 88))
        target_[:, min_key:max_key + 1] = np.asarray(save_est_logit).squeeze()
        save_est_logit = target_
        # save both Roll predictions and logits as npz files
        np.savez(f'{est_roll_folder}/' + str(start) + '-' + str(end) + '.npz', logit=save_est_logit, roll=save_est_roll)
        i = i+50

def torch_preprocess(input_file_list, label):
    input_img_list = []
    for input_file in input_file_list:
        input_img = Image.open(input_file).convert('L')
        binarr = np.array(input_img)
        input_img = Image.fromarray(binarr.astype(np.uint8))
        input_img_list.append(input_img)
    new_input_img_list = []
    for input_img in input_img_list:
        new_input_img_list.append(transform(input_img))
    final_input_img = np.concatenate(new_input_img_list)
    torch_input_img = torch.from_numpy(final_input_img).float().cuda()
    torch_label = torch.from_numpy(label).float().cuda()
    return torch_input_img, torch_label


if __name__ == "__main__":
    model_path = './models/BCE_5f_FAN.pth' # change to your path
    device = torch.device('cuda')
    net = Video2RollNet.resnet18()
    net.cuda()
    net.load_state_dict(torch.load(model_path))

    training_data = False
    # infer Roll predictions
    if training_data:
        for i in range(1, 25):
            video_name = f'Bach Prelude and Fugue No.{i} B1'
            img_folder = img_root + '/training/'+ video_name
            label_folder = label_root + '/training/'+ video_name+'.pkl'
            midi_folder = midi_root + '/training/'+ video_name
            est_roll_folder = est_roll_root + '/training/'+ video_name
            print("save file in:", est_roll_folder)
            os.makedirs(est_roll_folder, exist_ok=True)
            intervals, data = load_data(img_folder, label_folder, midi_folder)
            print("starting inference--------------------")
            inference(net,intervals, data, est_roll_folder)
    else:
        for i in range(1, 4):
            video_name = f'Bach Prelude and Fugue No.{i} B2'
            img_folder = img_root + '/testing/' + video_name
            label_folder = label_root + '/testing/' + video_name + '.pkl'
            midi_folder = midi_root + '/testing/' + video_name
            est_roll_folder = est_roll_root + '/testing/' + video_name
            print("save file in:", est_roll_folder)
            os.makedirs(est_roll_folder, exist_ok=True)
            intervals, data = load_data(img_folder, label_folder, midi_folder)
            print("starting inference--------------------")
            inference(net, intervals, data, est_roll_folder)

