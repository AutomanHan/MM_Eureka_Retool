import re
import pdb


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
        r"(.*?<code>.*?</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    )
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
        r"(.*?<code>.*?</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    pattern = r"(.*?<code>.*?</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    matches2 = re.search(pattern, completion, re.DOTALL)

    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    matches_res = re.findall(pattern_interpreter, completion, re.DOTALL)
    import pdb;pdb.set_trace()
    return matches, matches2
    return 0.5 if matches else 0.0

completion =  "<think>\nThe problem provides a triangle ABC with AD as its median and E as the midpoint of AD. The area of triangle ABE is given as 2. We need to find the area of triangle ABC.\n\nSince AD is the median of triangle ABC, it divides triangle ABC into two triangles with equal areas, i.e., Area(△ABD) = Area(△ACD). Also, Area(△ABC) = Area(△ABD) + Area(△ACD) = 2 * Area(△ABD).\n\nSince E is the midpoint of AD, BE divides triangle ABD into two triangles △ABE and △BDE. These two triangles share the same height from vertex B to the line containing AD. The bases of these triangles along the line AD are AE and DE. Since E is the midpoint of AD, AE = DE. Therefore, the areas of these two triangles are equal: Area(△ABE) = Area(△BDE).\n\nWe are given that Area(△ABE) = 2.\nSince Area(△ABE) = Area(△BDE), we have Area(△BDE) = 2.\nThe area of triangle ABD is the sum of the areas of triangle ABE and triangle BDE.\n\n<code>\n```python\n# Calculate the area of triangle ABD\narea_abe = 2\narea_bde = area_abe  # Area(△BDE) = Area(△ABE) since AE = DE and height from B is common\narea_abd = area_abe + area_bde\nprint(f\"Area(△ABD): {area_abd}\")\n```\n</code>\n<interpreter>\nArea(△ABD): 4\n</interpreter>\n\nThe area of triangle ABD is 4.\nSince AD is the median of triangle ABC, Area(△ABD) = Area(△ACD).\n\n<code>\n```python\n# Calculate the area of triangle ACD\narea_abd = 4\narea_acd = area_abd  # Area(△ACD) = Area(△ABD) since AD is the median\nprint(f\"Area(△ACD): {area_acd}\")\n```\n</code>\n<interpreter>\nArea(△ACD): 4\n</interpreter>\n\nThe area of triangle ACD is 4.\nThe area of triangle ABC is the sum of the areas of triangle ABD and triangle ACD.\n\n<code>\n```python\n# Calculate the area of triangle ABC\narea_abd = 4\narea_acd = 4\narea_abc = area_abd + area_acd\nprint(f\"Area(△ABC): {area_abc}\")\n```\n</code>\n<interpreter>\nArea(△ABC): 8\n</interpreter>\n\nThe area of triangle ABC is 8.\n</think>\n<answer>\nGiven that AD is the median of triangle ABC, it divides the triangle into two areas of equal size, meaning Area(△ABD) = Area(△ACD).\nE is the midpoint of AD. In triangle ABD, consider the line segment BE. Triangles ABE and BDE share the same height from vertex B to the line containing AD. Since E is the midpoint of AD, their bases AE and DE are equal. Thus, Area(△ABE) = Area(△BDE).\nGiven Area(△ABE) = 2, it follows that Area(△BDE) = 2.\nThe area of triangle ABD is the sum of the areas of triangles ABE and BDE: Area(△ABD) = Area(△ABE) + Area(△BDE) = 2 + 2 = 4.\nSince Area(△ABD) = Area(△ACD), Area(△ACD) = 4.\nThe area of triangle ABC is the sum of the areas of triangles ABD and ACD: Area(△ABC) = Area(△ABD) + Area(△ACD) = 4 + 4 = 8.\n\nThe final answer is $\\boxed{8}$.\n</answer>"
matches,matches2=format_reward_func(completion)
import pdb;pdb.set_trace()
print(matches)
print(matches2)