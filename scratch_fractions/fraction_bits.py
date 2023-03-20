import numpy as np

def get_string_from_double(num) -> str:
    parts = num.hex().split('0x1.')[-1].split('p')
    hex_mantissa = '1' + parts[0]
    unshifted = bin(int(hex_mantissa,16))[2:]
    place = int(parts[1])
    if place > 0:
        unshifted = unshifted + place*'0'
    elif place < 0:
        unshifted = (-1*place)*'0' + unshifted
    return unshifted.ljust(64, '0')

def as_array(num):
    string = get_string_from_double(num)
    bit_list = [int(x) for x in string]
    arr = np.array(bit_list)
    hot_bits = np.nonzero(arr)[0]
    return bit_list, hot_bits


def main():
    half = 0.5
    third = 1.0/3.0
    sixth = 1.0/6.0

    print(get_string_from_double(half))
    print(get_string_from_double(sixth))
    print(get_string_from_double(third))

    sum_list = None
    index_set = set()
    for value in [half, third, sixth]:
        (bits, indexes) = as_array(value)
        if sum_list is None:
            sum_list = bits
        else:
            sum_list = [x + y for x,y in zip(sum_list,bits)]
        for i in indexes:
            if i in index_set:
                print(f'Trying to add duplicate index: {i}')
            index_set.add(i)
    print(f'Zeros: {sum_list.count(0)}')
    print(f'Ones: {sum_list.count(1)}')
    print(f'Total = {sum_list.count(0)+sum_list.count(1)}')
    print(f'Number of elements: {len(sum_list)}')
    print(f'Number of elements in index set: {len(index_set)}')
    print(f'Minimum of elements in index set: {min(index_set)}')
    print(f'Maximum of elements in index set: {max(index_set)}')


if __name__ == '__main__':
    main()
