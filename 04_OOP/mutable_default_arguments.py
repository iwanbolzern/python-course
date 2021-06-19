def append_to(element, list_in=[]):
    list_in.append(element)
    return list_in


if __name__ == '__main__':
    my_list = append_to(12)
    print(my_list)

    my_other_list = append_to(42)
    print(my_other_list)
