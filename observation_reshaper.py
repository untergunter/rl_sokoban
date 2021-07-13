import numpy as np

def TransformToTiny(image,state):
    image_x, image_y,z = image.shape
    real_x, real_y = state.shape
    assert z == 3
    x_per_square = image_x / real_x
    y_per_square = image_y / real_y
    assert x_per_square == int(x_per_square)
    assert y_per_square == int(y_per_square)
    x_per_square = int(x_per_square)
    y_per_square = int(y_per_square)


    def observation_reshaper(image_to_reshape):
        tow_d = image_to_reshape.mean(axis=2)
        smallest_2d = np.zeros(shape=(real_x,real_y),dtype=float)
        for x in range(real_x):
            x_start = x*x_per_square
            x_end = (x+1)*x_per_square
            for y in range(real_y):
                y_start = y * y_per_square
                y_end = (y + 1) * y_per_square
                arr_slice = tow_d[x_start:x_end,y_start:y_end]
                smallest_2d[x,y] = np.mean(arr_slice)
        return smallest_2d

    """ now turn the reduced image to state """
    reshaped = observation_reshaper(image)
    image_to_state_mapper = dict()

    for x in range(real_x):
        for y in range(real_y):
            transformed = reshaped[x,y]
            result = state[x,y]
            if transformed in image_to_state_mapper:
                assert image_to_state_mapper[transformed]==result
            else:
                image_to_state_mapper[transformed] = result

    def reshaper_normalizer(image):
        smaller = observation_reshaper(image)
        normalized_state = np.zeros(shape=smaller.shape)
        for x in range(real_x):
            for y in range(real_y):
                normalized_state[x,y] = image_to_state_mapper[reshaped[x,y]]
        return normalized_state

    return reshaper_normalizer