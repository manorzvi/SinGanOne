

class SinGanOne_Error(Exception):
    """
    Base class for SinGanOne errors
    """
    pass


class SinGanOne_NonTorchDevice(SinGanOne_Error):
    """
    Device is not of type 'torch.device'
    """

    def __init__(self, device):
        self.device = device

    def __str__(self):
        return f'device type provided: {type(self.device)}'


class SinGanOne_Dataset_Error(SinGanOne_Error):
    """
    Base class for SinGanOne dataset errors
    """
    pass


class SinGanOne_PathNotFound(SinGanOne_Dataset_Error):
    """
    Image path doesn't exist.
    """

    def __init__(self, img_path):
        self.img_path = img_path

    def __str__(self):
        return f'{self.img_path} does not exist'