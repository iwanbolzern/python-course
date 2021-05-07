class MyClass:
    pass


if __name__ == '__main__':
    # as expected by most other languages
    print(MyClass())

    # special to python
    print(MyClass)
    # changing the class (we are working on the class definition and NOT on an object!)
    MyClass.new_attr = 'heyClass'
    print(MyClass.new_attr)


class MyMetaclass(type):

    def __new__(cls, *args, **kwargs):
        """__new__ is the method called before __init__
        it's the method that creates the object and returns it
        while __init__ just initializes the object passed as parameter
        """
        print(f'MyMetaclass __new__ called')
        return super().__new__(cls, *args, **kwargs)


class MyOtherClass(metaclass=MyMetaclass):

    def __init__(self):
        print(f'MyOtherClass __init__ called')

    def __new__(cls, *args, **kwargs):
        print(f'MyOtherClass __new__ called')
        return super().__new__(cls, *args, **kwargs)


if __name__ == '__main__':
    my_other_object = MyOtherClass()
    _ = MyOtherClass()