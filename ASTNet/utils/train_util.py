

def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']

    if train:
        inputs = video[:-1]
        target = video[-1]
        return inputs, target
        # return video, video_name
    else:   # TODO: bo sung cho test
        return video, video_name