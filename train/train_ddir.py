import os
import sys
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score, accuracy_score

def save_checkpoint(args, model, optimizer, epoch, best_val_acc, checkpoint_path):
    torch.save({
        'args': args,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):

    if os.path.exists(checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"Checkpoint loaded: start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

        return start_epoch, best_val_acc
    
    return 0, 0

def train(args, model, opt, train_loader, valid_loader, test_subj_index):

    checkpoint_path = os.path.join('checkpoint', args.dataset, f'checkpoint_subj_{test_subj_index+1}.pth')

    start_epoch, best_val_acc = load_checkpoint(model, opt, checkpoint_path)

    # Initialize TensorBoard writer
    log_dir = os.path.join('log', args.dataset, args.date, f'subject_{test_subj_index+1}')
    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()  # Start timing the epoch
        train_loss = 0
        model.train()

        for x in train_loader:

            model.zero_grad()
            opt.zero_grad()

            loss, _ = model(x, epoch)
            
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        model.eval()
        acc_subj = 0
        acc_task = 0
        kappa_task = 0

        for x in valid_loader:
            truth_subj, truth_task = x[1], x[2]
            _, preds = model(x, start_epoch)
            pred_subj, pred_task = preds[0], preds[1]

            acc1 = accuracy_score(truth_subj.cpu().numpy(), pred_subj.cpu().numpy())
            acc2 = accuracy_score(truth_task.cpu().numpy(), pred_task.cpu().numpy())
            kappa = cohen_kappa_score(truth_task.cpu().numpy(), pred_task.cpu().numpy())

            acc_subj += acc1
            acc_task += acc2
            kappa_task += kappa

        val_acc = acc_task / len(valid_loader)
        val_kappa = kappa_task / len(valid_loader)
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_kappa = val_kappa
            save_checkpoint(args, model, opt, epoch, best_val_acc, checkpoint_path)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/Validation_Subject', acc_subj / len(valid_loader), epoch)
        writer.add_scalar('Accuracy/Validation_Task', val_acc, epoch)
        writer.add_scalar('Accuracy/Best_Valid_Acc', best_val_acc, epoch)
        writer.add_scalar('Accuracy/Best_Valid_Kappa', best_val_kappa, epoch)

        epoch_duration = time.time() - epoch_start_time  # Calculate epoch duration

        print(f'test_subj:{test_subj_index+1}\tepoch:{epoch}\ttrain_loss: {train_loss/len(train_loader):.4f}\t'
              f'test_subj_acc: {acc_subj/len(valid_loader):.4f}\ttest_task_acc: {val_acc:.4f}\t'
              f'best_val_acc: {best_val_acc:.4f}\tbest_val_kappa: {best_val_kappa:.4f}\tepoch_time: {epoch_duration:.2f}s')

    writer.close()
    return best_val_acc, best_val_kappa