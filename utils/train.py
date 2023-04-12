import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets.dataset import collate_fn
from tqdm import tqdm
import torch
from evaluation_metric import compute_eer
from .viz import visualize_logs
from .loss import build_loss_func, compute_loss
from torchsummary import summary

def trainer(model, train_dataset, val_dataset, optimizer, scheduler, epochs, batch_size=64, loss_opt='ce', exp_name="baseline", device='cpu'):
    # Make directory to save pre-trained weights
    os.makedirs(f"checkpoint/{exp_name}", exist_ok=True)

    # Summarize model
    input_size = train_dataset[0][0].shape
    summary(model, input_size, batch_size=batch_size)

    # Log dictionary
    log_dict = {
        "Loss" : {"training" : [], "validation" : []},
        "Mean accuracy" : {"training" : [], "validation" : []},
        "Real accuracy" : {"training" : [], "validation" : []},
        "Fake accuracy" : {"training" : [], "validation" : []}
    }

    # Build dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Build loss func
    if loss_opt in ['ce']:
        loss_func = build_loss_func(loss_opt=loss_opt, device=device)
    else:
        cls_num_dict = train_dataset.get_cls_num()
        train_dataset.log_cls_num(cls_num_dict)
        loss_func = build_loss_func(loss_opt=loss_opt, device=device, cls_num_list=cls_num_dict["label"])

    prev_eer = 1e8
    for epoch in range(epochs):
        loading = tqdm(enumerate(train_dataloader))
        train_loss, train_acc = 0.0, 0.0
        real_acc, fake_acc = 0.0, 0.0
        print(f"Training ... [Epoch {epoch+1}/{epochs}]")
        model.train()
        for i, (inputs, targets) in loading:
            inputs, labels = inputs.to(device), targets['label'].to(device)

            # Calculate output
            features, probs = model(inputs)

            # Zero grad optimizer
            optimizer.zero_grad()
            loss = compute_loss(loss_func, probs, labels)
            loss.backward()

            # Gradient descent
            optimizer.step()

            # Log informations
            _, prediction = torch.max(probs, axis=1)
            train_loss += loss.item()
            train_acc += float(torch.sum(torch.eq(prediction, labels)))/len(prediction)

            # Print accuracies
            try:
                real_acc += float(torch.sum(torch.eq(prediction[labels == 0], labels[labels == 0])))/(len(prediction[labels == 0]))
            except:
                print("There is no real data on the current batch")
            
            try:
                fake_acc += float(torch.sum(torch.eq(prediction[labels == 1], labels[labels == 1])))/(len(prediction[labels == 1]))
            except:
                print("There is no fake data on the current batch")

            loading.set_description(f"Loss : {train_loss/(i+1):.4f}, Acc : {100*train_acc/(i+1):.4f}% \
                    (Real : {100*real_acc/(i+1):.4f}%, Fake : {100*fake_acc/(i+1):.4f}%)")
        
        # Append log information
        log_dict["Loss"]["training"].append(train_loss/len(train_dataloader))
        log_dict["Mean accuracy"]["training"].append(train_acc/len(train_dataloader))
        log_dict["Real accuracy"]["training"].append(real_acc/len(train_dataloader))
        log_dict["Fake accuracy"]["training"].append(fake_acc/len(train_dataloader))
        scheduler.step()

        print(f"Validating ... [Epoch {epoch+1}/{epochs}]")
        model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            loading = tqdm(enumerate(val_dataloader))
            val_loss, val_acc = 0.0, 0.0
            real_acc, fake_acc = 0.0, 0.0
            for i, (inputs, targets) in loading:
                inputs, labels = inputs.to(device), targets['label'].to(device)

                # Calculate output
                features, probs = model(inputs)
                loss = compute_loss(loss_func, probs, labels)

                # Log informations
                score = F.softmax(probs, dim=1)[:, 0]
                idx_loader.append(labels)
                score_loader.append(score)

                _, prediction = torch.max(probs, axis=1)
                val_loss += loss.item()
                val_acc += float(torch.sum(torch.eq(prediction, labels)))/len(prediction)

                # Print accuracies
                try:
                    real_acc += float(torch.sum(torch.eq(prediction[labels == 0], labels[labels == 0])))/(len(prediction[labels == 0]))
                except:
                    print("There is no real data on the current batch")
                
                try:
                    fake_acc += float(torch.sum(torch.eq(prediction[labels == 1], labels[labels == 1])))/(len(prediction[labels == 1]))
                except:
                    print("There is no fake data on the current batch")
                    
                loading.set_description(f"Loss : {val_loss/(i+1):.4f}, Acc : {100*val_acc/(i+1):.4f}% \
                    (Real : {100*real_acc/(i+1):.4f}%, Fake : {100*fake_acc/(i+1):.4f}%)")

            # Append log information
            log_dict["Loss"]["validation"].append(val_loss/len(val_dataloader))
            log_dict["Mean accuracy"]["validation"].append(val_acc/len(val_dataloader))
            log_dict["Real accuracy"]["validation"].append(val_acc/len(val_dataloader))
            log_dict["Fake accuracy"]["validation"].append(val_acc/len(val_dataloader))

            # Calculate EER
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_val_eer = compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            val_eer = min(val_eer, other_val_eer)
            print("Val EER: {}".format(val_eer))
            torch.save(model, os.path.join(f"checkpoint/{exp_name}", 'recent_model.pt'))
            if val_eer < prev_eer:
                # Save the model checkpoint
                torch.save(model, os.path.join(f"checkpoint/{exp_name}", 'best_model.pt'))
                prev_eer = val_eer
        
        # Plot log graph and save file
        visualize_logs(log_dict, exp_name=exp_name)
