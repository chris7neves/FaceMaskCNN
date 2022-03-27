import torch

def test(model, dataloaders, criterion):
    """
    The main testing loop for the model.
    :model: the model loaded with the parameters for its weights.
    :dataloaders: the dict containing the dataloaders that will be used for testing
    """

    testloader = dataloaders["test"]
    model.eval()
    with torch.no_grad(): # Doing torch.no_grad as well as putting model into eval mode might be redundant
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        for data in testloader:
            images, labels = data
            images = images.float()
            labels = labels.long()

            preds = model(images)
            preds = torch.argmax(preds, 1)

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

        return (all_labels, all_preds)