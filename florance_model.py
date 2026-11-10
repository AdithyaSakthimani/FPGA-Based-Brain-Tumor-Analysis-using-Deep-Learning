import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

class FlorenceModel:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            torch_dtype=self.dtype
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        )

    def describe(self, image_path , prompt):

        image = Image.open(image_path).convert("RGB")

        prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
    k: (
        v.to(self.device, dtype=self.dtype)
        if v.dtype.is_floating_point
        else v.to(self.device)
    )
    for k, v in inputs.items()
}

        with torch.no_grad():

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return generated_text