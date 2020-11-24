from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='kukkik-finetuned-checkpoint',
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch_finetuned.midi',
        prompt=None)
    
    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/continuation_finetuned.midi',
        prompt='./data/evaluation/012.midi')
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
