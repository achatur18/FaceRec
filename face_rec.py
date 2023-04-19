from extras import *
from model_zoo.model_zoo import PickableInferenceSession


class FaceRec:
    model_loc = "/content/"

    provider_options = get_default_provider_options()

    kwargs={
        "provider_options" : get_default_provider_options()
    }

    retinaface_onnx_file=model_loc+"det_10g.onnx"
    session = PickableInferenceSession(retinaface_onnx_file, **kwargs)
    retinaface=get_any_model(model_loc+"det_10g.onnx")

    arcface_onnx_file=model_loc+"w600k_r50.onnx"
    arcface_session = PickableInferenceSession(arcface_onnx_file, **kwargs)
    arcface=ArcFaceONNX(model_file=arcface_onnx_file, session=arcface_session)

    def get_embeddings(self, image_loc):
        img = ins_get_image(image_loc)
        input_shape=img.shape
        # input_shape=(2571, 2000, 3)
        bboxes, kpss = self.retinaface.detect(img,input_size=( 640, 640), max_num=0,metric='default')

        # if bboxes.shape[0] == 0:
        #     return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            # for taskname, model in self.models.items():
            #     model=recognition
            #     if taskname=='detection':
            #         continue
            self.arcface.get(img, face)
            ret.append(face)
        return ret