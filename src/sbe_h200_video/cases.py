from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class VideoCase:
    case_id: str
    split: str
    block_type: str
    prompt: str
    target: str
    negative: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


ATTRIBUTE = [
    ("emerald green scarf on a wooden chair", "emerald green scarf", "missing scarf or wrong color"),
    ("crimson teapot on a kitchen counter", "crimson teapot", "missing teapot or wrong color"),
    ("navy blue notebook on a white desk", "navy blue notebook", "missing notebook or wrong color"),
    ("golden bell on a dark shelf", "golden bell", "missing bell or wrong color"),
    ("pink soap bar beside a sink", "pink soap bar", "missing soap or wrong color"),
    ("black wallet on a marble table", "black wallet", "missing wallet or wrong color"),
    ("white ceramic rabbit figurine", "white rabbit figurine", "missing figurine or wrong color"),
    ("turquoise water bottle on a desk", "turquoise water bottle", "missing bottle or wrong color"),
    ("orange safety helmet on concrete", "orange safety helmet", "missing helmet or wrong color"),
    ("purple gift box on a table", "purple gift box", "missing box or wrong color"),
    ("red wool hat on a bench", "red wool hat", "missing hat or wrong color"),
    ("blue glass vase on a shelf", "blue glass vase", "missing vase or wrong color"),
    ("yellow raincoat hanging on a hook", "yellow raincoat", "missing raincoat or wrong color"),
    ("green ceramic bowl on a table", "green ceramic bowl", "missing bowl or wrong color"),
    ("silver spoon on a napkin", "silver spoon", "missing spoon or wrong material"),
    ("brown leather shoe on a rug", "brown leather shoe", "missing shoe or wrong color"),
    ("white cotton towel folded on a shelf", "white cotton towel", "missing towel or wrong color"),
    ("orange traffic cone on asphalt", "orange traffic cone", "missing cone or wrong color"),
    ("black camera body on a tripod", "black camera body", "missing camera or wrong color"),
    ("red apple on a white plate", "red apple on white plate", "missing apple or wrong color"),
]

CONTAINMENT = [
    ("silver ring inside a small wooden box", "ring inside wooden box", "ring outside box or missing box"),
    ("red marble inside a transparent cup", "red marble inside cup", "marble outside cup or missing cup"),
    ("blue toy boat inside a glass bowl", "toy boat inside bowl", "boat outside bowl or missing bowl"),
    ("green leaf inside a clear jar", "green leaf inside jar", "leaf outside jar or missing jar"),
    ("yellow sponge inside a metal bucket", "sponge inside bucket", "sponge outside bucket or missing bucket"),
    ("small white shell inside a blue dish", "shell inside blue dish", "shell outside dish or missing dish"),
    ("brown acorn inside a white mug", "acorn inside mug", "acorn outside mug or missing mug"),
    ("gray stone inside a red tray", "stone inside tray", "stone outside tray or missing tray"),
    ("orange ball inside a wire basket", "orange ball inside basket", "ball outside basket or missing basket"),
    ("purple bead inside a clear bottle", "purple bead inside bottle", "bead outside bottle or missing bottle"),
    ("green toy car inside a cardboard box", "toy car inside box", "toy car outside box"),
    ("yellow lemon inside a glass cup", "lemon inside glass cup", "lemon outside cup"),
    ("white dice inside a black bowl", "dice inside bowl", "dice outside bowl"),
    ("red flower inside a transparent vase", "flower inside vase", "flower outside vase"),
    ("blue pen inside a pencil case", "pen inside pencil case", "pen outside case"),
    ("small key inside a ceramic dish", "key inside dish", "key outside dish"),
    ("pink eraser inside a plastic container", "eraser inside container", "eraser outside container"),
    ("gray pebble inside a clear glass", "pebble inside glass", "pebble outside glass"),
    ("toy duck inside a bathtub", "toy duck inside bathtub", "duck outside bathtub"),
    ("red cube inside a transparent box", "red cube inside transparent box", "cube outside box"),
]

PRESENCE = [
    ("single umbrella standing in an empty hallway", "one umbrella visible", "missing umbrella"),
    ("single bicycle parked beside a brick wall", "one bicycle visible", "missing bicycle"),
    ("single guitar leaning against a sofa", "one guitar visible", "missing guitar"),
    ("single clock hanging on a plain wall", "one clock visible", "missing clock"),
    ("single camera on a tripod in a studio", "camera on tripod visible", "missing camera or tripod"),
    ("single plant pot on a window ledge", "plant pot visible", "missing plant pot"),
    ("single pair of sunglasses on a table", "sunglasses visible", "missing sunglasses"),
    ("single desk lamp on a nightstand", "desk lamp visible", "missing desk lamp"),
    ("single red suitcase in a hotel room", "suitcase visible", "missing suitcase"),
    ("single laptop on a clean desk", "laptop visible", "missing laptop"),
    ("single skateboard against a wall", "skateboard visible", "missing skateboard"),
    ("single violin on a chair", "violin visible", "missing violin"),
    ("single microwave in a kitchen", "microwave visible", "missing microwave"),
    ("single teacup on a saucer", "teacup visible", "missing teacup"),
    ("single baseball glove on grass", "baseball glove visible", "missing glove"),
    ("single candle on a dark table", "candle visible", "missing candle"),
    ("single watering can in a garden", "watering can visible", "missing watering can"),
    ("single backpack near a door", "backpack visible", "missing backpack"),
    ("single remote control on a couch", "remote control visible", "missing remote"),
    ("single white chair in an empty room", "chair visible", "missing chair"),
]

SPATIAL = [
    ("red bowl to the left of a white cup", "red bowl left of white cup", "wrong left-right relation"),
    ("green book to the right of a black phone", "green book right of black phone", "wrong left-right relation"),
    ("blue cube above a yellow cube", "blue cube above yellow cube", "wrong vertical relation"),
    ("pink eraser below a silver pencil", "pink eraser below silver pencil", "wrong vertical relation"),
    ("orange ball in front of a blue box", "orange ball in front of blue box", "wrong depth relation"),
    ("white mug behind a red plate", "white mug behind red plate", "wrong depth relation"),
    ("black shoe beside a green backpack", "black shoe beside green backpack", "not beside or missing object"),
    ("yellow lemon between two red apples", "lemon between two apples", "wrong between relation"),
    ("purple cup left of a silver spoon", "purple cup left of silver spoon", "wrong left-right relation"),
    ("blue bottle right of a white bowl", "blue bottle right of white bowl", "wrong right relation"),
    ("red cube above a green cylinder", "red cube above green cylinder", "wrong above relation"),
    ("small key below a black wallet", "key below wallet", "wrong below relation"),
    ("orange cone in front of a gray rock", "cone in front of rock", "wrong front relation"),
    ("white candle behind a blue vase", "candle behind vase", "wrong behind relation"),
    ("green pear between two yellow bananas", "pear between bananas", "wrong between relation"),
    ("silver laptop beside a red mouse", "laptop beside mouse", "not beside"),
    ("black camera left of a brown bag", "camera left of bag", "wrong left relation"),
    ("blue mug right of a white plate", "mug right of plate", "wrong right relation"),
    ("yellow toy truck in front of a red block", "truck in front of block", "wrong front relation"),
    ("green plant behind a small lamp", "plant behind lamp", "wrong behind relation"),
]

STATE = [
    ("open blue lunchbox on a table", "open blue lunchbox", "closed lunchbox or missing lunchbox"),
    ("closed green book on a desk", "closed green book", "open book or missing book"),
    ("lit red candle on a dark table", "lit red candle", "unlit candle or missing flame"),
    ("folded white towel on a shelf", "folded white towel", "unfolded towel or missing towel"),
    ("inflated beach ball on the sand", "inflated beach ball", "deflated ball or missing ball"),
    ("empty transparent glass on a counter", "empty transparent glass", "filled glass or missing glass"),
    ("peeled banana on a plate", "peeled banana", "unpeeled banana or missing banana"),
    ("zipped black backpack on the floor", "zipped black backpack", "unzipped backpack or missing backpack"),
    ("open silver laptop on a desk", "open laptop", "closed laptop"),
    ("closed red umbrella on the floor", "closed umbrella", "open umbrella"),
    ("turned-on desk lamp on a table", "lamp turned on", "lamp turned off"),
    ("unlit white candle on a shelf", "unlit candle", "lit candle"),
    ("broken egg on a plate", "broken egg", "unbroken egg"),
    ("full glass of water on a table", "full glass", "empty glass"),
    ("tied shoelace on a black shoe", "tied shoelace", "untied shoelace"),
    ("unwrapped chocolate bar on paper", "unwrapped chocolate bar", "wrapped chocolate"),
    ("open cardboard box on the floor", "open cardboard box", "closed box"),
    ("closed suitcase near a bed", "closed suitcase", "open suitcase"),
    ("burning match on a dark background", "burning match", "unlit match"),
    ("deflated soccer ball on grass", "deflated soccer ball", "inflated soccer ball"),
]

COUNTING = [
    ("exactly two silver spoons on a napkin", "exactly two spoons", "wrong number of spoons"),
    ("exactly three red cherries on a plate", "exactly three cherries", "wrong number of cherries"),
    ("exactly four blue buttons on white cloth", "exactly four buttons", "wrong number of buttons"),
    ("exactly two yellow toy ducks in a row", "exactly two ducks", "wrong number of ducks"),
    ("exactly three black pens on a desk", "exactly three pens", "wrong number of pens"),
    ("exactly four green blocks on the floor", "exactly four blocks", "wrong number of blocks"),
    ("exactly two white candles on a shelf", "exactly two candles", "wrong number of candles"),
    ("exactly three purple cups on a table", "exactly three cups", "wrong number of cups"),
    ("exactly two red apples in a bowl", "exactly two apples", "wrong number of apples"),
    ("exactly three blue balls on a carpet", "exactly three balls", "wrong number of balls"),
    ("exactly four yellow lemons on a plate", "exactly four lemons", "wrong number of lemons"),
    ("exactly two green bottles on a shelf", "exactly two bottles", "wrong number of bottles"),
    ("exactly three orange cones on a road", "exactly three cones", "wrong number of cones"),
    ("exactly four white dice on a table", "exactly four dice", "wrong number of dice"),
    ("exactly two black shoes beside a door", "exactly two shoes", "wrong number of shoes"),
    ("exactly three silver keys on a counter", "exactly three keys", "wrong number of keys"),
    ("exactly four pink erasers on paper", "exactly four erasers", "wrong number of erasers"),
    ("exactly two brown cookies on a plate", "exactly two cookies", "wrong number of cookies"),
    ("exactly three green pears in a row", "exactly three pears", "wrong number of pears"),
    ("exactly four red cubes on a white surface", "exactly four red cubes", "wrong number of cubes"),
]


BLOCKS = {
    "object attribute": ATTRIBUTE,
    "containment": CONTAINMENT,
    "object presence": PRESENCE,
    "spatial relation": SPATIAL,
    "object state": STATE,
    "counting": COUNTING,
}


def build_cases() -> list[VideoCase]:
    cases: list[VideoCase] = []
    for block_type, items in BLOCKS.items():
        for idx, (phrase, target, negative) in enumerate(items):
            split = "dev" if idx < 6 else "heldout"
            case_id = f"{block_type.replace(' ', '_')}_{idx:02d}"
            prompt = (
                f"A realistic ten second video of {phrase}, steady camera, "
                "clear view, natural lighting."
            )
            cases.append(VideoCase(case_id, split, block_type, prompt, target, negative))
    return cases


def select_cases(split: str, max_prompts: int) -> list[dict[str, str]]:
    cases = [c for c in build_cases() if split == "all" or c.split == split]
    # Balanced round-robin by block type.
    buckets = {block_type: [c for c in cases if c.block_type == block_type] for block_type in BLOCKS}
    selected: list[VideoCase] = []
    max_bucket = max((len(items) for items in buckets.values()), default=0)
    for idx in range(max_bucket):
        for block_type in BLOCKS:
            items = buckets[block_type]
            if idx < len(items):
                selected.append(items[idx])
                if len(selected) >= max_prompts:
                    return [c.to_dict() for c in selected]
    return [c.to_dict() for c in selected[:max_prompts]]


def block_types() -> Iterable[str]:
    return BLOCKS.keys()
