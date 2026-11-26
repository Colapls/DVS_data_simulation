"""三类别目标映射表：CARLA 蓝图 ID → car / motorcycle / pedestrian。

映射表按《DVS 自动标注方案（精简可执行版）》§6.3 显式列出，不做子串猜测。
CARLA 的摩托车蓝图是 ``vehicle.kawasaki.ninja`` 这类 ID，既不含 "motorcycle"
也不含 "bicycle" 子串，按子串匹配必然把摩托车标成汽车（缺陷 D2）。

表外的新蓝图不会被静默归类：``vehicle.*`` 里没有显式条目的按"其余车辆 → car"
兜底，但会带上 ``by_fallback=True``，采集器把它写进 ``unknown_blueprints.csv``，
由人决定是加入表格还是加入忽略名单；不认识的命名空间直接记 ``unknown``。
"""

from __future__ import annotations

from typing import Any


#: 映射表版本；改动本文件里的清单时同步递增，数据集里会带上这个版本号。
CLASS_MAP_VERSION = "2.0"

#: 类别名 → category_id（写入标注文件，加载器可直接使用）。
CLASSES: dict[str, int] = {"car": 0, "motorcycle": 1, "pedestrian": 2}

#: category_id → 类别名。
CATEGORY_NAMES: tuple[str, ...] = tuple(CLASSES)

#: CARLA 0.9.16 的摩托车蓝图（显式清单）。
MOTORCYCLE_BLUEPRINTS = frozenset(
    {
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.vespa.zx125",
    }
)

#: 自行车不属于三类别：生成时剔除，偶尔出现也只记 ignore。
BICYCLE_BLUEPRINTS = frozenset(
    {
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    }
)

#: 明确不属于三类别、但仍会出现在世界里的车辆资产（拖挂等）。
IGNORED_BLUEPRINTS = frozenset({"vehicle.carlamotors.carlacola"})

#: 车系子类关键词：只用于人工排查和统计，不参与类别判定。
SUBTYPE_RULES: tuple[tuple[str, str], ...] = (
    ("ambulance", "emergency"),
    ("police", "emergency"),
    ("firetruck", "emergency"),
    ("bus", "bus"),
    ("truck", "truck"),
    ("van", "van"),
    ("suv", "suv"),
    ("offroad", "suv"),
    ("wrangler", "suv"),
    ("cybertruck", "suv"),
    ("sedan", "sedan"),
)

#: 目标轮廓占 3D 框投影面积的经验比例：visibility = 轮廓占比 / 填充率。
#: 行人体型细长，填充率明显低于车辆；不同 CARLA 版本可按需调整。
FILL_RATIO: dict[str, float] = {
    "car": 0.55,
    "motorcycle": 0.45,
    "pedestrian": 0.30,
}


def _subtype(blueprint_id: str) -> str:
    """按蓝图 ID 猜车系子类；猜不出来记 ``other``（不是类别名 ``car``）。"""
    for keyword, subtype in SUBTYPE_RULES:
        if keyword in blueprint_id:
            return subtype
    return "other"


def classify_blueprint(blueprint_id: str) -> dict[str, Any]:
    """把蓝图 ID 映射成类别信息。

    返回 ``{"category", "category_id", "subtype", "ignore_reason", "by_fallback"}``：

    - ``category`` ∈ {car, motorcycle, pedestrian, ignore, unknown}
    - ``ignore`` = 明确不属于三类别（自行车、拖挂），写入 ``ignored_annotations``
    - ``unknown`` = 不认识的命名空间；采集器跳过并记入 unknown_blueprints.csv
    - ``by_fallback`` = 该类别靠"其余 vehicle.* → car"的兜底规则得到，
      没有显式条目；采集器会把这类蓝图列出来，禁止静默归类
    """
    blueprint = blueprint_id.lower()
    if blueprint.startswith("walker.pedestrian"):
        return {
            "category": "pedestrian",
            "category_id": CLASSES["pedestrian"],
            "subtype": "pedestrian",
            "ignore_reason": None,
            "by_fallback": False,
        }
    if blueprint in MOTORCYCLE_BLUEPRINTS:
        return {
            "category": "motorcycle",
            "category_id": CLASSES["motorcycle"],
            "subtype": "motorcycle",
            "ignore_reason": None,
            "by_fallback": False,
        }
    if blueprint in BICYCLE_BLUEPRINTS:
        return {
            "category": "ignore",
            "category_id": None,
            "subtype": "bicycle",
            "ignore_reason": "bicycle",
            "by_fallback": False,
        }
    if blueprint in IGNORED_BLUEPRINTS:
        return {
            "category": "ignore",
            "category_id": None,
            "subtype": None,
            "ignore_reason": "not_in_target_classes",
            "by_fallback": False,
        }
    if blueprint.startswith("vehicle."):
        return {
            "category": "car",
            "category_id": CLASSES["car"],
            "subtype": _subtype(blueprint),
            "ignore_reason": None,
            "by_fallback": True,
        }
    return {
        "category": "unknown",
        "category_id": None,
        "subtype": None,
        "ignore_reason": "unknown_blueprint",
        "by_fallback": False,
    }


def is_target_blueprint(blueprint_id: str) -> bool:
    """是否属于要标注的三类别目标（生成交通流时用来剔除自行车）。"""
    return classify_blueprint(blueprint_id)["category"] in CLASSES
