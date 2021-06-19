class Vehicle(object):
    """A generic vehicle class."""

    def __init__(self, position):
        self.position = position

    def travel(self, destination):
        print('Travelling...')


class RadioMixin(object):
    def play_song_on_station(self, station):
        print('Playing station...')


class Car(Vehicle, RadioMixin):
    pass


class Boat(Vehicle):
    pass
