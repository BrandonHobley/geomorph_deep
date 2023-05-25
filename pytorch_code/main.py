from architectures import seg_arcs
import os
import tqdm
import torch
import torchvision
import re
import torch.nn as nn
import numpy as np
import scipy.io as sio
from datasets import geomorph_split, geomorph_all
from sklearn.metrics import confusion_matrix

LEARNING_RATE = 0.001
BATCH_SIZE = 12
NUM_EPOCHS = 300

# WEIGHTS FOR TRAINING. THE WEIGHTS WERE COMPUTED BASED ON PIXEL COUNT FREQUENCY FOR EACH LABEL CLASS.

# 1 - COMPUTE PIXEL COUNTS FOR EACH LABEL AND DIVIDE BY TOTAL NUMBER OF LABELLED PIXELS.
# 2 - THEN, COMPUTE PER-CLASS PROBABILITY BASED ON PIXEL COUNTS
#   - class_probability = per_class_pixel_count ./ total_labelled_pixels
# 3 - USE PROBABILITIES TO COMPUTE THE WEIGHT FOR EACH CLASS DURING TRAINING. AND NORMALISE BY NUMBER OF CLASSES
#   - class_weights = 1 ./ (class_probability .* NUM_CLASSES)
w = torch.Tensor([9.61278783154613,
                3.01992944742642,
                3.39024883162814,
                3.45438867438867,
                2.31779401007802,
                1.34857455297769,
                1.01080653593613,
                3.21645977400413,
                0.170364396474057,
                1.56877847189512]).cuda()

# TEST SET PIXEL COUNT. THIS MAY BE DIFFERENT DEPENDING ON THE TRAIN/TEST SPLIT
pc = [3507,
7570,
14161,
7258,
18119,
7407,
76975,
26933,
498023,
18643]

# CHANNEL MEANS AND VARIANCES IN 0-1 DOUBLE PRECISION RANGE.
means = [0.623653852981452,	0.619295127782880,	0.820967395296985,	0.610019588006864,	0.455613046054422,	0.563967388543578,	0.419404337636785,	0.5, 0.896254483455195,	0.896261500081612]
stds = [0.117874868848855,	0.112586722174056,	0.135060313047434,	0.0746555779645883,	0.0526050325699416,	0.0668346203407143,	0.0474701783893535,	0.5, 0.0934844694472531,	0.0934937786429269]


def update_ema_variables(model, ema_model, alpha=0.99):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    return model, ema_model


def generate_loaders():

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])

    print('building loaders...')

    train_set = geomorph_split.GeomorphIR(root=r'DIR_TO_ROOT_FROM_MATLAB_PRE-PROCESS', split='train',
                                        transform=transforms, target_transform=None)

    test_set = geomorph_split.GeomorphIR(root=r'DIR_TO_ROOT_FROM_MATLAB_PRE-PROCESS', split='test',
                                       transform=transforms, target_transform=None)


    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    return train_set, test_set, train_loader, test_loader


def stich_preds():

    model_dir = r'DIR_TO_MODEL'
    preds_dir = r'OUTPUT_DIR_TO_TILED_PREDS'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])

    print('building loaders...')

    test_set = geomorph_all.GeomorphIR(root=r'DIR_TO_ROOT_FROM_MATLAB_PRE-PROCESS', transform=transforms,
                                     target_transform=None)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    model = seg_arcs.ResSegUBN_t(input_channs=6, output_channs=10).cuda()
    #model = seg_arcs.SegU_t(input_channs=6, output_channs=10).cuda()

    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model.eval()
    torch.cuda.empty_cache()

    for step, (img, img_path) in enumerate(test_loader):
        print('----------------------------------------------------------------')
        with torch.no_grad():

            [l_coords, split] = split_image(img)
            coord_idx = 0
            count = 0

            for b_img in split:
                b_img = b_img.cuda()

                out = model(b_img)
                _, predicted = torch.max(out.data, 1)
                np_predicted = predicted.cpu().detach().numpy()
                coords = l_coords[coord_idx]

                s = img_path[0].split("\\")
                f = s[-1].replace('.mat', '')

                im_coords = list(map(int, re.findall(r'\d+', f)))

                out_dir = str(im_coords[0]) + '_' + str(im_coords[1])
                out_dir = os.path.join(preds_dir, out_dir)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_file = str(coords[0]) + '_' + str(coords[1]) + '.mat'

                count += 1
                coord_idx += 1

                out_file = os.path.join(out_dir, out_file)
                print(out_file)

                sio.savemat(out_file, {'preds': np_predicted})

    return


def cmts_process():

    model_root = r'DIR_TO_MODELS'
    cmt_root = r'DIR_TO_TEST_SET_MAT_FILES'

    N = 88 # NUMBER OF TEST SET IMAGES
    accs = []
    for f in os.listdir(cmt_root):
        curr_dir = os.path.join(cmt_root, f)
        print(curr_dir)
        p_cat = []
        y_cat = []
        class_acs = np.zeros((10, 1))
        for idx_ in range(0, N):

            p = os.path.join(curr_dir, 'p_' + str(idx_) + '.mat')
            y = os.path.join(curr_dir, 'y_' + str(idx_) + '.mat')
            pred = sio.loadmat(os.path.join(curr_dir, p))
            pred = pred['p']
            for el in pred:
                p_cat.append(el)

            label = sio.loadmat(os.path.join(curr_dir, y))
            label = label['y']
            for el in label:
                y_cat.append(el)
        p_cat = np.concatenate(p_cat, axis=0)
        y_cat = np.concatenate(y_cat, axis=0)
        cmat = confusion_matrix(y_cat, p_cat)
        [r, c] = cmat.shape
        for i in range(r):
            for j in range(c):
                if i == j:
                    h = cmat[i, j]
                    class_acs[i] = h / pc[i]
        acc = np.mean(class_acs)
        accs.append(acc)

    accs = - np.array(accs)
    sorted_idxs = np.argsort(accs)
    for i in range(5):
        print(os.path.join(model_root, (os.listdir(cmt_root)[sorted_idxs[i]])))
        print(np.abs(accs[sorted_idxs[i]]))

    exit(-1)

    idx = accs.index(max(accs))
    best_acc = max(accs)
    best_model_dir = os.path.join(model_root, (os.listdir(cmt_root)[idx]))

    return best_model_dir, best_acc


def train(train_loader):

    root = r'OUTPUT_DIR_FOR_SAVED_MODELS'

    #model = seg_arcs.ResSegUBN_t(input_channs=6, output_channs=10).cuda()
    model = seg_arcs.SegU_t(input_channs=6, output_channs=10).cuda()
    #model_T = seg_arcs.ResSegUBN_t(input_channs=10, output_channs=10).cuda()
    #model_T.load_state_dict(model.state_dict())
    #for param in model_T.parameters():
    #    param.requires_grad = False
    #    param.detach_()

    # OPTIM, LOSS and GAUSSIAN SMOOTHING
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(w, ignore_index=-1, reduction='mean')

    train_losses_epoch = []

    for epoch in range(0, NUM_EPOCHS):

        train_losses = []
        str_ep = str(epoch)

        for i, (image, label) in tqdm.tqdm(enumerate(train_loader)):

            x = image.cuda()
            y = label.cuda()

            # sup loss
            pred_logits_S = model(x)
            loss = criterion(pred_logits_S, y)

            # unsup loss
            #with torch.no_grad():
            #    pred_logits_T = model_T(x).detach()

            # Logits to probs and consistency
            #pred_prob_T = F.softmax(pred_logits_T, dim=1)
            #pred_nll_S = -F.log_softmax(pred_logits_S, dim=1)
            #loss2 = (pred_nll_S * pred_prob_T).sum(dim=1)
            #pred_conf_t, _ = torch.max(pred_prob_T, dim=1)
            #pred_conf_mask = (pred_conf_t >= 0.98).float().mean()
            #loss2 = (loss2 * (y == -1).float()) * pred_conf_mask * 0.1

            #loss = loss1 + loss2.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            #update_ema_variables(model, model_T)

            if torch.isnan(loss) == 0:
                train_losses.append(loss.item())

        # EPOCH TRAINING LOSS
        avg_train_loss = sum(train_losses) / len(train_losses)


        # CHECK IF CURRENT MODEL LOSS IS BETTER
        if ((epoch > 0) and (avg_train_loss < min(train_losses_epoch))):
            print('---------------------------------------------------------------------')
            print('Train loss at epoch {}: {}'.format(epoch, avg_train_loss))
            print('Previous best train loss : {}'.format(min(train_losses_epoch)))

            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'loss': avg_train_loss}
            out_file = 'bb_sup_EPS_' + str_ep + '_UNET.pth'
            out = os.path.join(root, out_file)
            torch.save(state, out)

        # SAVE EVERY 20 EPOCHS
        if epoch % 20 == 0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'loss': avg_train_loss}
            out_file = 'bb_sup_EPS_' + str_ep + '_UNET.pth'
            out = os.path.join(root, out_file)
            torch.save(state, out)

        # LOSS CHECK EVERY 10 EPOCHS
        if epoch % 10 == 0:
            print('=====================================================================')
            print('EPOCH: {}'.format(epoch + 1))
            print('CURRENT TRAIN LOSS: {}'.format(avg_train_loss))
            #print('CURRENT UNSUPERVISED LOSS: {}'.format(loss2.mean()))

        train_losses_epoch.append(avg_train_loss)


def test_fold(test_loader):

    model_root = r'DIR_TO_MODELS'
    cmt_root = r'OUTPUT_DIR_FOR_TEST_SET_PREDICTIONS'

    #model = seg_arcs.ResSegUBN_t(input_channs=6, output_channs=10).cuda()
    model = seg_arcs.SegU_t(input_channs=6, output_channs=10).cuda()

    for f in os.listdir(model_root):
        curr_model = os.path.join(model_root, f)
        model.load_state_dict(torch.load(curr_model)['state_dict'])
        model.eval()
        cmt_out = os.path.join(cmt_root, os.path.splitext(os.path.basename(curr_model))[0])
        print(cmt_out)
        if not(os.path.exists(cmt_out)):
            os.mkdir(cmt_out)

        with torch.no_grad():
            for i, (image, label) in tqdm.tqdm((enumerate(test_loader))):
                x = image.cuda()
                y = label.cuda()  # add dim for transformaction


                m = y != -1

                out = model(x)
                _, p = torch.max(out.data, 1)

                p = p.view(-1)
                y = y.view(-1)
                m = m.view(-1)

                # DISCARD NON-LABELLED PIXELS
                p = p[m].detach().cpu().numpy()
                y = y[m].detach().cpu().numpy()

                p_f = 'p_' + str(i) + '.mat'
                y_f = 'y_' + str(i) + '.mat'

                # SAVES AS .MAT FILE
                p_out = os.path.join(cmt_out, p_f)
                y_out = os.path.join(cmt_out, y_f)

                sio.savemat(p_out, {'p': p})
                sio.savemat(y_out, {'y': y})

    return None


if __name__ == '__main__':

    #stich_preds()
    _, _, train_loader, test_loader = generate_loaders()
    train(train_loader)
    test_fold(test_loader)
    model_dir, acc = cmts_process()