import kaldiio
import matplotlib.pyplot as plt
import argparse
import os


# load scp file
def load_scp(scp_dict, o_dir):
    for key in scp_dict:
        out_f = os.path.join(o_dir, key)
        mel = scp_dict[key].T[::-1]
        print(mel)
        plt.imshow(mel)
        plt.title(key)
        #plt.colorbar()
        plt.savefig(out_f)
        plt.clf()


# load ark file
def load_ark(ark_generator, o_dir):
    for key, array in ark_generator:
        out_f = os.path.join(o_dir, key)
        plt.imshow(array.T[::-1])
        plt.title(key)
        #plt.colorbar()
        plt.savefig(out_f)
        plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", type=str, default=None)
    parser.add_argument("--ark", type=str, default="/home/Data/program_data/espnet2/dump/char_dev/feats.1.ark")
    parser.add_argument("--odir", type=str, default="temp_scp")
    args = parser.parse_args()
    if not os.path.isdir(args.odir):
        os.system("mkdir {}".format(args.odir))

    if args.scp is not None:
        scp_dict = kaldiio.load_scp(args.scp)
        load_scp(scp_dict, args.odir)
    elif args.ark is not None:
        ark_generator = kaldiio.load_ark(args.ark)
        load_ark(ark_generator, args.odir)
    else:
        pass


if __name__ == '__main__':
    #scp_f = "/home/Data/program_data/espnet2/dump/char_dev/feats.1.scp"
    #ark_f = "/home/Data/program_data/espnet2/dump/char_dev/feats.1.ark"
    scp_f = "/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/ref/fbank/raw_fbank_data.1.scp"
    ark_f = "/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/ref/fbank/raw_fbank_data.1.ark"

    ark_f = "/home/Data/program_data/espnet2/dump/char_dev/feats.1.ark:26"

    out_d_scp = "temp_scp"
    out_d_ark = "temp_ark"
    main()