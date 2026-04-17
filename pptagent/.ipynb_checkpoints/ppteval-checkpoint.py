import json
import os
from collections import defaultdict
from glob import glob

from jinja2 import Template
from llms import LLM
from tqdm import tqdm

from pptagent.presentation import Presentation
from pptagent.utils import Config, package_join, pdirname, pexists
import func_argparse
language_model = LLM(os.environ.get("LANGUAGE_MODEL", "gpt-4.1"), os.environ.get("API_BASE", None))
vision_model = LLM(os.environ.get("LANGUAGE_MODEL", "gpt-4.1"), os.environ.get("API_BASE", None))


text_scorer = Template(
    open(package_join("prompts", "ppteval_content.txt")).read()
)
vision_scorer = Template(
    open(package_join("prompts", "ppteval_style.txt")).read()
)
style_descriptor = open(
    package_join("prompts", "ppteval_describe_style.txt")
).read()
content_descriptor = open(
    package_join("prompts", "ppteval_describe_content.txt")
).read()
ppt_extractor = Template(
    open(package_join("prompts", "ppteval_extract.txt")).read()
)
logic_scorer = Template(
    open(package_join("prompts", "ppteval_coherence.txt")).read()
)


def get_eval(prs_source: str):
    evals = defaultdict(dict)
    eval_file = package_join(pdirname(prs_source), "evals.json")
    if pexists(eval_file):
        with open(eval_file) as f:
            evals |= json.load(f)
    return evals, eval_file


def slide_score(prs_source: str, slide_folder: str):
    evals, eval_file = get_eval(prs_source)
    for slide_image in glob(package_join(slide_folder, "slide_*.jpg")) + glob(
        package_join(slide_folder, "slide_images", "slide_*.jpg")
    ):
        slide_descr = slide_image.replace(".jpg", ".json")
        if not os.path.exists(slide_descr):
            style_descr = vision_model(style_descriptor, slide_image)
            content_descr = vision_model(content_descriptor, slide_image)
            with open(slide_descr, "w") as f:
                json.dump(
                    {"content": content_descr, "style": style_descr},
                    f,
                    indent=2,
                )
        else:
            descr = json.load(open(slide_descr))
            style_descr = descr["style"]
            content_descr = descr["content"]
        if slide_image not in evals["vision"]:
            evals["vision"][slide_image] = language_model(
                vision_scorer.render(descr=style_descr), return_json=True
            )
        if slide_image not in evals["content"]:
            evals["content"][slide_image] = language_model(
                text_scorer.render(descr=content_descr), return_json=True
            )
    with open(eval_file, "w") as f:
        json.dump(evals, f, indent=2)


def pres_score(prs_source: str):
    slide_folder = pdirname(prs_source)
    tmp_config = Config(slide_folder)
    evals, eval_file = get_eval(prs_source)
    if "logic" in evals:
        return
    slide_descr = package_join(slide_folder, "extracted.json")
    if not pexists(slide_descr):
        presentation = Presentation.from_file(prs_source, tmp_config).to_text()
        extracted = language_model(
            ppt_extractor.render(presentation=presentation),
            return_json=True,
        )
        with open(slide_descr, "w") as f:
            json.dump(extracted, f, indent=2)
    else:
        extracted = json.load(open(slide_descr))
    evals["logic"] = language_model(
        logic_scorer.render(
            presentation=extracted,
        ),
        return_json=True,
    )
    with open(eval_file, "w") as f:
        json.dump(evals, f, indent=2)


def eval_ppt(prs_files: str, slide_folders: str):
    pres_score(prs_files)
    
    slide_score(prs_files, slide_folders)

if __name__ == "__main__":
    func_argparse.main(eval_ppt)