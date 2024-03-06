import numpy as np

class Dialog:
    def __init__(self, utterances, labels, speakers, features):
        self.utterances = utterances
        self.labels = labels
        self.speakers = speakers
        self.features = features
        self.numberofemotionshifts = 0
        self.numberofspeakers = 0
        self.numberofutterances = 0
        self.difficulty = 0
        self.cc()

    def __getitem__(self, item):
        if item == 'utterances':
            return self.utterances
        elif item == 'labels':
            return self.labels
        elif item == 'speakers':
            return self.speakers
        elif item == 'features':
            return self.features
    #measure the difficulty of a dialog
    def cc(self):
        neg_list = [2, 3, 5, 6]
        pos_list = [1, 4]
#        {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        self.numberofutterances = len(self.utterances)
        speaker_emo = {}
        for i in range(0, len(self.labels)):
            if (self.speakers[i] in speaker_emo):
                speaker_emo[self.speakers[i]].append(self.labels[i])
            else:
                speaker_emo[self.speakers[i]] = [self.labels[i]]
        for key in speaker_emo:
            for i in range(0, len(speaker_emo[key]) - 1):
                if speaker_emo[key][i] != speaker_emo[key][i+1]:
                    self.numberofemotionshifts += 1
        self.numberofspeakers = len(set(self.speakers))
        self.difficulty = (self.numberofemotionshifts + self.numberofspeakers) / (self.numberofutterances + self.numberofspeakers)