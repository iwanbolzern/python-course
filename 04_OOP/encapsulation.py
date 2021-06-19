class Person:
    def __init__(self, first_name: str, last_name: str) -> None:
        self._first_name = first_name  # underscore suggests that the field is private
        self._last_name = last_name

    def print(self) -> None:
        print(f"My name is {self._first_name} {self._last_name}.")


class PrivatePerson:
    def __init__(self, first_name: str, last_name: str) -> None:
        self.__first_name = first_name  # double underscore makes the field private
        self.__last_name = last_name

    def print(self) -> None:
        print(f"My name is {self.__first_name} {self.__last_name}.")


class PropertyPerson:
    def __init__(self, first_name: str, last_name: str) -> None:
        self.first_name = first_name
        self.last_name = last_name

    @property
    def first_name(self) -> str:
        return self.__first_name

    @first_name.setter
    def first_name(self, name: str) -> None:
        self.__first_name = name

    @property
    def last_name(self) -> str:
        return self.__last_name

    @last_name.setter
    def last_name(self, name: str) -> None:
        self.__last_name = name

    @property
    def full_name(self) -> str:
        return f"{self.__first_name} {self.__last_name}"

    @full_name.setter
    def full_name(self, name: str) -> None:
        first, last = name.split()  # for simplicity assume there's always exactly one space
        self.__first_name = first
        self.__last_name = last

    def print(self) -> None:
        print(f"My name is {self.full_name}.")


if __name__ == '__main__':
    print("Using single underscore:")
    me = Person("Paola", "Bianchi")
    me.print()
    me._first_name = "Iwan"
    me._last_name = "Bolzern"
    me.print()

    print("\nUsing double underscore:")
    me = PrivatePerson("Paola", "Bianchi")
    me.print()
    # print(me.__first_name)  # AttributeError
    print(me.__dict__)

    print("\nUsing properties:")
    me = PropertyPerson("Paola", "Bianchi")
    print(me.__dict__)
    me.print()
    me.full_name = "Iwan Bolzern"
    print(me.__dict__)
    me.print()

