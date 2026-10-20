"""Landmark knowledge base loaded from special_buildings_en.csv."""
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
        """All landmarks of a town."""
        return self.by_town.get(town, [])

    def get_landmark_types(self, town):
        """Distinct building types of a town."""
        types = sorted(set(item["building_type"] for item in self.by_town.get(town, [])))
        return types

    def find_coordinate(self, town, building_type, index=0):
        """The index-th building of a type (the first if index is out of range)."""
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
        """The building of a type nearest to (ref_x, ref_y)."""
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
