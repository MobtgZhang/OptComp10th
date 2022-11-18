import numpy as np
def tflite_inference(model, img):
    """Inferences an image through the model with tflite interpreter on CPU
    :param model: a tflite.Interpreter loaded with a model
    :param img: image
    :return: list of outputs of the model
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.resize_tensor_input(input_details[0]["index"], img.shape)
    model.allocate_tensors()
    model.set_tensor(input_details[0]["index"], img.astype(np.float32))
    model.invoke()
    return [model.get_tensor(elem["index"]) for elem in output_details]
class StageStatus:
    """Keeps status between MTCNN stages"""

    def __init__(self, pad_result: tuple = None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmp_w = self.tmp_h = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmp_w, s.tmp_h = pad_result