def make_batchset(
        data_json,
        sort_key,
        batch_size,
        batch_max
):
    """
    data_json = {
        "utt1": {
          "utt2spk": "Tom",
          "input": [
                          {
                              "feat": "/Users/rosen/speech_project/tts/espnet/egs/blizzard13/tts2_gst/dump/char_train_no_dev/feats.1.ark:29",
                              "name": "input1",
                              "shape": [
                                  968,
                                  80]
                          },
                          {
                              "feat": "/Users/rosen/speech_project/tts/espnet/egs/blizzard13/tts2_gst/dump/char_train_no_dev/feats.4.ark:29",
                              "name": "input2",
                              "shape": [
                                  968,
                                  513]
                          }
                      ],
          "output": [
                    {
                    "name": "target1",
                    "shape": [108,42],
                    "text": "JANE EYRE AN AUTOBIOGRAPHY BY CHARLOTTE BRONTE CHAPTER I THERE WAS NO POSSIBILITY OF TAKING A WALK THAT DAY.",
                    "token": "J A N E <space> E Y R E <space> A N <space> A U T O B I O G R A P H Y <space> B Y <space> C H A R L O T T E <space> B R O N T E <space> C H A P T E R <space> I <space> T H E R E <space> W A S <space> N O <space> P O S S I B I L I T Y <space> O F <space> T A K I N G <space> A <space> W A L K <space> T H A T <space> D A Y .",
                    "tokenid": "19 10 23 14 8 14 34 27 14 8 10 23 8 10 30 29 24 11 18 24 16 27 10 25 17 34 8 11 34 8 12 17 10 27 21 24 29 29 14 8 11 27 24 23 29 14 8 12 17 10 25 29 14 27 8 18 8 29 17 14 27 14 8 32 10 28 8 23 24 8 25 24 28 28 18 11 18 21 18 29 34 8 24 15 8 29 10 20 18 23 16 8 10 8 32 10 21 20 8 29 17 10 29 8 13 10 34 6"
                    }]
                  }
      },
      {
        "utt2": {
        ...
      },
        ...
      }

    sort_key : auto, input, output

    return List[List[Tuple[str, dict]]]
    [
        [("utt1": {input:[{feats:...,}, {feats:...,}], output:[{text:...,]}),
        ("utt2": ...
        ...
        ], # batch1
    ]
    """
    batches = []
    # sort by keys
    data_json_tuple = list(zip(data_json.keys(), data_json.values()))  # to (str, dict)
    for k, v in data_json_tuple:
        print(k, v)
    data_json_tuple.sort(key=lambda x: x[1][sort_key][0]["shape"][0])

    # Filter out Sequence longer than threshold
    data_json_tuple_flt = list(filter(lambda x: x[1][sort_key][0]["shape"][0] < batch_max, data_json_tuple))
    # got batch list
    for batch in get_batch(data_json_tuple_flt, batch_size):
        batches.append(batch)
    return batches


def get_batch(data, batch_size=3):
    """
    """
    l = len(data)
    for i in range(0, l, batch_size):
        yield data[i: min(i + batch_size, l)]