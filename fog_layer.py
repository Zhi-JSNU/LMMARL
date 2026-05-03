import random
from typing import List, Tuple

class GroundBaseStation:
    def __init__(self, station_id, x, y):
        self.station_id = station_id
        self.x = x
        self.y = y
        self.z = 0
        self.computing_capability_ghz = 25.0
        self.cycles_per_bit = random.randint(500, 1500)

    def __str__(self):
        return (f"Ground BS {self.station_id} | "
                f"Pos: ({self.x}, {self.y}, {self.z}) | "
                f"Computing: {self.computing_capability_ghz:.2f} GHz | "
                f"Cycles per bit(C_B): {self.cycles_per_bit} cycles/bit")

class FogLayer:

    def __init__(self, station_positions: List[Tuple[float, float]]):
        self.base_stations = []
        self._create_all_base_stations(station_positions)

    def _create_all_base_stations(self, station_positions: List[Tuple[float, float]]):
        for i, (x, y) in enumerate(station_positions):
            station = GroundBaseStation(station_id=f"B{i+1}", x=x, y=y)
            self.base_stations.append(station)

    def display_status(self):
        print(f"\n--- Fog Computing Layer Status ({len(self.base_stations)} base stations) ---")
        for station in self.base_stations:
            print(station)
