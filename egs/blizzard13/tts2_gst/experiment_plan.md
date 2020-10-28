1. Selection Test
    - Pitch
    - Energy
    -



2. Transfer Test
    - Attention map
    - Happy/Sad/Normal

Short: 20 chars
Long : 100 chars

Ref_audio(Open_sameSPK/Open_diffSPK):
    happy_long
    happy_short

    sad_long
    sad_short

    normal_long
    normal_short



Text:
    Short
    Long

| Ref_audio     | Text     | Header Two     |
| :------------- | :------------- |:------------- |
| Happy_long       |  Long        | Long       |
| Sad_long       |  Long        | Long       |
| Happy_short       |  Long        | Long       |
| Sad_short       |  Long        | Long       |
| normal_long       |  Long        | Long       |
| normal_long       |  Long        | Long       |


- Does happy correctly transfered?

To do List:

1. Vocoder (stft => wav)
2. Salient attention<1>
3. Emotion detection(select audio)
4. Decoding problem in run.sh (code is wrong, since syn.sh is good) <2>

5. Audio feature related to emotion
    - duration
    - pitch
    - energy

6. Explore data
    - Reduce training data to 3~5 arts (100h <-> 1500*128*4/3600=213h)
    - 
7. 