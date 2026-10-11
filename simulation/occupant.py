"""
Occupant Agent Module: Simulates realistic daily human routines, room occupancy,
and natural movement within the smart home.
"""
import math
import random


class Occupant:
    """Represents a resident living in the smart home."""

    def __init__(self, occupant_id, name, color, home_room_id, schedule_type="alex"):
        self.occupant_id = occupant_id
        self.name = name
        self.color = color                # RGB tuple for avatar
        self.home_room_id = home_room_id
        self.schedule_type = schedule_type

        # Spatial position in normalized floorplan coordinates (0.0 to 1.0)
        self.pos = [0.2, 0.6]
        self.target_pos = [0.2, 0.6]
        self.current_room_id = home_room_id
        self.current_activity = "Sleeping"
        self.is_home = True
        self.move_speed = 0.15  # Speed in normalized units per real second
        self.walk_phase = 0.0

    def get_scheduled_activity(self, sim_hour):
        """
        Returns (target_room_id, activity_name, is_home) based on 24-hr schedule.
        """
        if self.schedule_type == "alex":
            # Alex's Schedule (Working Professional)
            if 0.0 <= sim_hour < 6.5:
                return "master_bed", "Sleeping", True
            elif 6.5 <= sim_hour < 7.25:
                return "bathroom", "Morning Shower", True
            elif 7.25 <= sim_hour < 8.15:
                return "kitchen", "Breakfast Prep", True
            elif 8.15 <= sim_hour < 8.4:
                return "foyer", "Leaving for Work", True
            elif 8.4 <= sim_hour < 17.75:
                return "outside", "Away at Office", False
            elif 17.75 <= sim_hour < 18.1:
                return "foyer", "Returning Home", True
            elif 18.1 <= sim_hour < 20.25:
                return "living_room", "Relaxing & Watching TV", True
            elif 20.25 <= sim_hour < 21.25:
                return "kitchen", "Dinner Time", True
            elif 21.25 <= sim_hour < 22.75:
                return "living_room", "Evening Leisure", True
            elif 22.75 <= sim_hour < 23.4:
                return "bathroom", "Night Routine", True
            else:
                return "master_bed", "Preparing to Sleep", True

        else:
            # Sam's Schedule (Remote Worker / Student)
            if 0.0 <= sim_hour < 7.25:
                return "study", "Sleeping", True
            elif 7.25 <= sim_hour < 8.0:
                return "bathroom", "Morning Routine", True
            elif 8.0 <= sim_hour < 8.75:
                return "kitchen", "Coffee & Breakfast", True
            elif 8.75 <= sim_hour < 12.5:
                return "study", "Work / Coding on PC", True
            elif 12.5 <= sim_hour < 13.5:
                return "kitchen", "Lunch Break", True
            elif 13.5 <= sim_hour < 17.0:
                return "study", "Afternoon Work", True
            elif 17.0 <= sim_hour < 18.5:
                return "living_room", "Tea & Relax", True
            elif 18.5 <= sim_hour < 20.5:
                return "living_room", "TV & Entertainment", True
            elif 20.5 <= sim_hour < 21.5:
                return "kitchen", "Dinner", True
            elif 21.5 <= sim_hour < 22.75:
                return "study", "Reading & Gaming", True
            elif 22.75 <= sim_hour < 23.3:
                return "bathroom", "Bedtime Prep", True
            else:
                return "study", "Sleeping", True

    def get_room_anchor(self, room_id, house):
        """Returns a suitable target position inside the specified room."""
        if room_id == "outside":
            return [-0.1, 0.2]  # Offscreen position

        room = house.rooms.get(room_id)
        if not room:
            return [0.5, 0.5]

        rx, ry, rw, rh = room.bounds
        # Add slight personality offsets so occupants don't overlap exactly
        offset_x = 0.35 if self.schedule_type == "alex" else 0.65
        offset_y = 0.55 if self.schedule_type == "alex" else 0.45

        return [rx + rw * offset_x, ry + rh * offset_y]

    def update(self, real_dt, sim_dt, sim_hour, house):
        """
        Updates occupant routine, target room, and movement interpolation.
        """
        target_room_id, activity, is_home = self.get_scheduled_activity(sim_hour)
        self.current_activity = activity
        self.is_home = is_home

        # Determine target coordinates
        target_pos = self.get_room_anchor(target_room_id, house)
        self.target_pos = target_pos

        # Smooth interpolation towards target position
        dx = self.target_pos[0] - self.pos[0]
        dy = self.target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist > 0.01:
            # Step size based on frame time
            step = min(dist, self.move_speed * real_dt * 2.5)
            self.pos[0] += (dx / dist) * step
            self.pos[1] += (dy / dist) * step
            self.walk_phase += real_dt * 10.0
        else:
            self.pos[0] = self.target_pos[0]
            self.pos[1] = self.target_pos[1]

        # Update room occupancy registration in House
        old_room_id = self.current_room_id
        if is_home:
            current_room = house.get_room_for_pos(self.pos[0], self.pos[1])
            self.current_room_id = current_room.room_id if current_room else target_room_id
        else:
            self.current_room_id = "outside"

        # Sync with House room occupant lists
        for room in house.rooms.values():
            if self in room.occupants and room.room_id != self.current_room_id:
                room.occupants.remove(self)
            elif self not in room.occupants and room.room_id == self.current_room_id:
                room.occupants.append(self)

        # Realistic Human Appliance Interactions (Manual Mode fallback)
        self._apply_human_device_interactions(house, sim_hour)

        return old_room_id != self.current_room_id

    def _apply_human_device_interactions(self, house, sim_hour):
        """Simulates humans naturally operating appliances based on their current activity."""
        is_night_sleep = (sim_hour >= 22.5 or sim_hour < 6.5)

        # Master bedroom occupant (Alex) sleeping routine
        if self.schedule_type == "alex":
            bed_room = house.rooms.get("master_bed")
            if bed_room:
                for d in bed_room.devices:
                    if d.name == "Master Split AC (1.5T)":
                        if is_night_sleep and self.current_room_id == "master_bed":
                            if not d.is_on and not d.manual_override:
                                d.set_state(True)
                                d.target_temp = 23.5
                        elif not is_night_sleep and self.current_room_id != "master_bed":
                            if d.is_on and not d.manual_override and not bed_room.is_occupied:
                                d.set_state(False)
