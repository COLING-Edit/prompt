from model_args import get_args
opt=get_args()
if opt.task=='sentiment':
    if opt.setting=='zero-shot':
        prefix="The sentiment of the text {"
        postfix="} is: "

    else: # few-shot
        prefix = """
                Sentence: This movie is very exciting.
                Sentiment: {positive}

                ####

                Sentence: I hate this movie, it sucks.
                Sentiment: {negative}

                ####

                Sentence: This movie was actually pretty funny.
                Sentiment: {positive}

                ####

                """
        adding1 = \
            '''Sentence: i can't believe how tedious this movie is...
            Sentiment: {negative}

            ####

            Sentence: This restaurant has good service and delicious food!
            Sentiment: {positive}

            ####
            '''
        adding2 = \
            '''
            Sentence: i would never recommend anyone to live in here 
            Sentiment: {negative}

            ####

            Sentence: even in summer, they have decent patronage.
            Sentiment: {positive}

            ####

            '''
        if opt.setting=='3-shot':prefix=prefix
        elif opt.setting=='5-shot':
            prefix=prefix+adding1
        elif opt.setting=='7-shot':
            prefix=prefix+adding1+adding2
        postfix = "Sentiment: {"

    stopwords='#$%&()*+,-–./:;<=>@[\\]^_`{|}~—•…�'+'0123456789'#+'bcdefghjklmnopqrstvwxyz'

elif opt.task=='formality':
    prefix = """
    Sentence: i do not intend to be mean
    Formality: {formal}

    #####

    Sentence: ohhh i don't intend to be mean ..
    Formality: {informal}

    #####

    Sentence: what 're u doing here ?
    Formality: {informal}

    #####
    
    Sentence: people are having coffee , lunch, playing in the park , playing and talking .
    Formality: {formal}

    #####
    
    Sentence: well, that is simply the manner it is done, i suppose.
    Formality: {formal}
    
    ####
    
    Sentence: well that is just the way it is I guess.
    Formality: {informal}
    
    ####
    
    Sentence: hello, i am in NYC and i could assist you if you need.
    Formality: {formal}

    """
    postfix="Formality: {"
    #stopwords = '#$%&()*+,-–./:;<=>@[\\]^_`{|}~•…�'+'0123456789'
    stopwords = '#$%|~…�<=>'
