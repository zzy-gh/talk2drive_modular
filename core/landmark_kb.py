"""
知识库模块: 加载special_buildings_en.csv, 提供给LLM解析模块和坐标查询模块使用。
"""
import csv
from collections import defaultdict


class LandmarkKnowledgeBase:
    def __init__(self, csv_path="special_buildings_en.csv"):
        self.by_town = defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.by_town[row["town"]].append({
                    "building_type": row["building_type"],
                    "carla_name": row["carla_name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                })

    def get_landmarks(self, town):
        """返回某个Town的全部地标(给LLM prompt用)"""
        return self.by_town.get(town, [])

    def get_landmark_types(self, town):
        """返回某个Town有哪些building_type(去重), 给LLM prompt用更简洁"""
        types = sorted(set(item["building_type"] for item in self.by_town.get(town, [])))
        return types

    def find_coordinate(self, town, building_type, index=0):
        """
        按town+building_type查真实坐标。
        如果同类型有多个(比如Town01有8个Supermarket), 默认返回第一个,
        index参数允许选择第几个(配合距离最近等逻辑可以扩展)。
        """
        candidates = [item for item in self.by_town.get(town, [])
                      if item["building_type"].lower() == building_type.lower()]
        if not candidates:
            return None
        if index >= len(candidates):
            index = 0
        return candidates[index]

    def find_all(self, town, building_type):
        """Return all buildings of a given type in a town."""
        return [item for item in self.by_town.get(town, [])
                if item["building_type"].lower() == building_type.lower()]

    def find_nearest_coordinate(self, town, building_type, ref_x, ref_y):
        """如果同类型建筑有多个, 返回离参考点(比如车辆当前位置)最近的那个"""
        candidates = [item for item in self.by_town.get(town, [])
                      if item["building_type"].lower() == building_type.lower()]
        if not candidates:
            return None
        def dist2(item):
            return (item["x"] - ref_x) ** 2 + (item["y"] - ref_y) ** 2
        return min(candidates, key=dist2)


if __name__ == "__main__":
    kb = LandmarkKnowledgeBase("special_buildings_en.csv")
    print("Town01 landmark types:", kb.get_landmark_types("Town01"))
    print("Town01 GasStation coord:", kb.find_coordinate("Town01", "GasStation"))
