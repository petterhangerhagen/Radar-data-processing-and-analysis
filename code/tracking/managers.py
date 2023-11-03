from copy import copy
import numpy as np

class Manager(object):
    def __init__(self, tracker, initiator, terminator, confirmation_threshold):
        self.tracker = tracker
        self.initiator = initiator
        self.terminator = terminator
        self.track_history = dict()
        self.tracks = set()
        self.preliminary_track_history = dict()
        self.preliminary_tracks = set()
        self.__confirmation_threshold__ = confirmation_threshold

    def step(self, measurements, timestamp, **kwargs):
        if 'ownship' in kwargs.keys():
            self.tracker.filter.measurement_model.set_ownship_position(kwargs['ownship'])

        tracks, unused_measurements = self.tracker.step(self.tracks | self.preliminary_tracks, measurements, timestamp)

        tracks = self.terminator.step(tracks)

        self.tracks, self.preliminary_tracks = self.__confirm_tracks__(tracks)

        self.preliminary_tracks = self.preliminary_tracks | self.initiator.step(unused_measurements, measurements=measurements, timestamp=timestamp)

        self.__update_track_history__()


    def __update_track_history__(self):
        for track in self.tracks:
            if track.index in self.track_history.keys():
                self.track_history[track.index].append(track)
            else:
                self.track_history[track.index] = [track]
        for track in self.preliminary_tracks:
            if track.index in self.preliminary_track_history.keys():
                self.preliminary_track_history[track.index].append(track)
            else:
                self.preliminary_track_history[track.index] = [track]

    def __confirm_tracks__(self, tracks):
        preliminary_tracks = set()
        confirmed_tracks = set()
        track_indices = [track.index for track in self.tracks]
        for track in tracks:
            if track.existence_probability < self.__confirmation_threshold__ and track.index not in track_indices:
                preliminary_tracks.add(track)
            elif track.existence_probability >= self.__confirmation_threshold__ and track.index not in track_indices:
                confirmed_tracks.add(track)
                self.track_history[track.index] = self.preliminary_track_history[track.index]
            else:
                confirmed_tracks.add(track)

        return confirmed_tracks, preliminary_tracks
