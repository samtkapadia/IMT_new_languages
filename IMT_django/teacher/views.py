###################################################################################################################
# Date: June 2015
# Author: Sam Kapadia (sam.kapadia.20@ucl.ac.uk)
###################################################################################################################


# Import some Django modules
import numpy as np
import os
import librosa
from scipy import spatial
import pickle
import speech_recognition as srr
import random
import unicodedata
import string
import datetime

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.http import JsonResponse
from django.shortcuts import redirect
from urllib.request import urlopen
from django.urls import reverse

# Import our eer module (Expected Error Reduction)
from . import eer
from . import helper_functions as helper

# Import the User and UserResponse models (which are stored in the SQL database)
from teacher.models import User, UserResponse


r = srr.Recognizer()

# Define the teaching and testing lengths
vocab_size = 100
num_teaching_images = 30
num_testing_images = 30

datasets_path = '/home/samkapadia/Datasets/'
userdata_path = '/home/samkapadia/User-Data/'

# Load the image paths to the 100 hf words
image_paths = list(np.load(os.path.join(datasets_path + 'image_paths_czech100.npy')))
audio_paths = list(np.load(os.path.join(datasets_path + 'audio_paths_czech100.npy')))

words = list(np.load(os.path.join(datasets_path + 'czech_words_100.npy')))
words = sorted(words, key=str.casefold)

test_words = list(np.load(os.path.join(datasets_path + 'czech_words_30_test.npy')))
test_words = [int(x) for x in test_words]

cs_en_dict_fp = os.path.join(datasets_path + 'cs_en_dict.pickle')
with open(cs_en_dict_fp, 'rb') as handle:
    cs_en_dict = pickle.load(handle)


def index(request):

    if request.method == 'GET':
        request.session.flush()

    if 'mode' in request.session:
        mode = int(request.session['mode'])

        if mode == -3:
            request.session['mode'] = -2
            return render(request, 'teacher/consent_form.html')

        if mode == -2:
            request.session['mode'] = -1
            return render(request, 'teacher/newuser.html')

        if mode == -1:
            request.session['mode'] = 0
            return render(request, 'teacher/mic_test.html')

        # Last mode was new user
        if mode == 0:
            createNewUser(request)
            request.session['teaching_image_num'] = 0
            return teaching(request)

        # Last mode was teaching
        elif mode == 1:
            return HttpResponse('Audio received')

        # Last mode was feedback
        elif mode == 2:
            teaching_image_num_ = int(request.session['teaching_image_num'])

            # If teaching is over
            if teaching_image_num_ == num_teaching_images:
                request.session['mode'] = 3
                context = {'num_testing_images': num_testing_images}
                return render(request, 'teacher/endteaching.html', context)
            else:
                # Continue teaching
                return teaching(request)

        # For testing
        elif mode == 3: # last mode was endTeaching
            request.session['testing_image_num'] = 0
            return testing(request)

        elif mode == 4: # last mode was testing
            testing_image_num_ = int(request.session['testing_image_num'])
            if testing_image_num_ == num_testing_images:
                return testResults(request)
            else:
                return testing(request)

    # No mode, therefore the user has just visited the website
    else:
        request.session['mode'] = -3
        return render(request, 'teacher/participant_info.html')



def get_random_string(length=10):

    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def createNewUser(request):


    # Create new user
    num_users = User.objects.count()
    user_id = num_users
    new_user = User.create(user_id)

    # assign user to a graph type ['BASE', 'WAV2VEC2', 'CONVAUTO', 'PHONEMES_CONCAT']
    num_graphs = []
    num_graphs.append(User.objects.filter(graph='BASE').count())
    num_graphs.append(User.objects.filter(graph='WAV2VEC2').count())
    num_graphs.append(User.objects.filter(graph='CONVAUTO').count())
    num_graphs.append(User.objects.filter(graph='PHONEMES_CONCAT').count())
    min_graph = np.argmin(num_graphs)
    user_graph = ['BASE', 'WAV2VEC2', 'CONVAUTO', 'PHONEMES_CONCAT'][min_graph]
    new_user.graph = user_graph

    # save user
    ucode_ = get_random_string()
    request.session['ucode'] = str(ucode_)
    new_user.ucode = str(ucode_)
    new_user.start_time = datetime.datetime.now()
    new_user.save()

    # Create X as an empty belief state (X is the machine's model of the student's distribution)
    X = np.zeros((vocab_size, 2))
    X_path = os.path.join(userdata_path + 'X_' + str(user_id) + '.npy')
    np.save(X_path, X)

    # Set L as an unlabelled set
    L = []
    request.session['L'] = L

    f_track = np.zeros((vocab_size, 2, num_teaching_images))
    f_track_path = userdata_path + 'F_track_' + str(user_id) + user_graph + '.npy'
    np.save(f_track_path, f_track)

    # Set up the session
    request.session['user_id'] = user_id
    request.session['testing_samples'] = test_words
    request.session['labelled_words'] = []
    request.session['tested_words'] = []
    request.session['mode'] = '0'
    request.session['graph'] = user_graph
    request.session['score'] = '0'
    request.session['test_scores'] = []
    request.session['responses'] = []


def teaching(request):

    user_id_ = request.session['user_id']
    teaching_image_num_ = request.session['teaching_image_num']
    testing_samples_ = request.session['testing_samples']
    graph_ = request.session['graph']

    teaching_image_num = teaching_image_num_ + 1

    W = np.load(os.path.join(datasets_path + 'weight_matrix_{}.npy'.format(graph_)))
    Y = np.load(os.path.join(datasets_path + 'ground_truth.npy'))
    X_path = userdata_path + 'X_' + str(user_id_) + '.npy'
    X = np.load(X_path)
    L = request.session['L']

    if request.session['graph'] == 'BASE':
        nS = W.shape[0]
        U = np.setdiff1d(np.arange(nS), L)
        available = np.setdiff1d(U, test_words)
        next_sample = int(random.choice(available))
    else:
        next_sample = int(eer.get_next_sample(X, Y, W, L, testing_samples_))

    image_path = image_paths[next_sample]

    context = {'teaching_image_num': teaching_image_num, 'num_teaching_images': num_teaching_images,
               'image_path': image_path}

    labelled_words_ = request.session['labelled_words']
    labelled_words_.append(words[next_sample])

    request.session['teaching_image_id'] = next_sample
    request.session['labelled_words'] = labelled_words_
    request.session['teaching_image_num'] = teaching_image_num
    request.session['image_path'] = image_path
    request.session['audio_path'] = audio_paths[next_sample]
    request.session['mode'] = 2

    # response server -> user front end
    return render(request, 'teacher/16khz-audio-recording.html', context)


def feedback(request):

    teaching_image_num_ = int(request.session['teaching_image_num'])
    image_path_ = request.session['image_path']
    audio_path_ = request.session['audio_path']

    is_correct = False
    score_ = request.session['score']

    if score_ == '1':
        is_correct = True

    labelled_word_ = request.session['labelled_words'][-1]
    english_word_ = cs_en_dict[labelled_word_]

    context = {'teaching_image_num': teaching_image_num_, 'image_path': image_path_, 'is_correct': is_correct,
               'audio_ref_link': audio_path_, 'translation': english_word_} #words_EN[next_sample]}

    request.session['mode'] = 2

    return render(request, 'teacher/feedback.html', context)


def testing(request):

    # initialises at 0
    testing_image_num_ = request.session['testing_image_num']
    testing_samples_ = request.session['testing_samples']

    # this means it starts at 1
    testing_image_num = testing_image_num_ + 1

    testing_image_id = testing_samples_[testing_image_num - 1]
    image_path = image_paths[testing_image_id]

    request.session['testing_image_num'] = testing_image_num
    request.session['testing_image_id'] = testing_image_id

    context = {'testing_image_num': testing_image_num, 'num_testing_images': num_testing_images,
                'image_path': image_path}

    return render(request, 'teacher/testing.html', context)


def processTeachingAnswer(request):

    user_id_ = int(request.session['user_id'])
    teaching_image_id_ = int(request.session['teaching_image_id'])
    teaching_image_num_ = request.session['teaching_image_num']
    word_ = request.session['labelled_words'][-1]

    tmp_filename = userdata_path + 'recordings/{}_teach{}_{}'.format(user_id_, teaching_image_num_, word_)
    with open('{}.wav'.format(tmp_filename), mode='bx') as f:
        f.write(request.body)

    audio_data_sr = srr.AudioFile('{}.wav'.format(tmp_filename))

    L_ = request.session['L']
    L_.append(teaching_image_id_)
    X_path = userdata_path + 'X_' + str(user_id_) + '.npy'
    X_ = np.load(X_path)

    word_norm = unicodedata.normalize('NFD', word_)
    word_conv = u"".join([c for c in word_norm if not unicodedata.combining(c)])

    with audio_data_sr as source:
        audio = r.record(source)
    try:
        transcription = r.recognize_google(audio, show_all=True, language="cs-CZ")['alternative'][0]['transcript']
    except:
        transcription = 'UNK'

    trans_norm = unicodedata.normalize('NFD', transcription)
    trans_conv = u"".join([c for c in trans_norm if not unicodedata.combining(c)])

    # google translate returns numbers - use written form instead
    numbers_dict = {'4': words[11], '44': words[12], '9': words[15],
                    '11': words[26], '6': words[78]}
    if trans_conv in list(numbers_dict.keys()):
        transcription = numbers_dict[trans_conv]
        trans_norm = unicodedata.normalize('NFD', transcription)
        trans_conv = u"".join([c for c in trans_norm if not unicodedata.combining(c)])

    X_[teaching_image_id_, :] = 0.0
    if trans_conv.lower() == word_conv.lower():
        pronunciation = 1
    else:
        pronunciation = 0

    X_[teaching_image_id_, pronunciation] = 1

    graph_ = request.session['graph']

    # TRACKING F ##############
    W = np.load(os.path.join(datasets_path + 'weight_matrix_{}.npy'.format(graph_)))
    Y = np.load(os.path.join(datasets_path + 'ground_truth.npy'))
    f_u, U = eer.get_f(X_, Y, W, L_)

    # transfer estimations to X
    for i, unlabelled in enumerate(U):
        X_[unlabelled, :] = f_u[i, :].copy()

    # store 'correct' column for this teaching round in f_track
    teaching_image_num_ = request.session['teaching_image_num']
    user_graph_ = request.session['graph']
    f_track_path = userdata_path + 'F_track_' + str(user_id_) + user_graph_ + '.npy'
    f_track = np.load(f_track_path)
    f_track[:, :, teaching_image_num_ - 1] = X_.copy()

    request.session['score'] = str(pronunciation)
    request.session['L'] = L_

    responses_ = request.session['responses']
    responses_.append('{}: {}'.format(word_, pronunciation))
    request.session['responses'] = responses_

    responses_path = os.path.join(userdata_path + 'responses_' + str(user_id_) + '.npy')
    np.save(responses_path, request.session['responses'])

    np.save(X_path, X_)
    np.save(f_track_path, f_track)

    perm_filename = userdata_path + 'recordings/{}_teach{}_{}_{}'.format(user_id_, teaching_image_num_, word_, pronunciation)
    with open('{}.wav'.format(perm_filename), mode='bx') as f:
        f.write(request.body)
    os.remove('{}.wav'.format(tmp_filename))

    return HttpResponse('transcription: {}, {}'.format(transcription, pronunciation))


def processTestingAnswer(request):

    user_id_ = int(request.session['user_id'])
    testing_image_num_ = request.session['testing_image_num']
    testing_samples_ = request.session['testing_samples']
    testing_image_id = testing_samples_[testing_image_num_ - 1]
    word_ = words[testing_image_id]

    tmp_filename = userdata_path + 'recordings/{}_test{}_{}'.format(user_id_, testing_image_num_, word_)
    with open('{}.wav'.format(tmp_filename), mode='bx') as f:
        f.write(request.body)

    audio_data_sr = srr.AudioFile('{}.wav'.format(tmp_filename))

    with audio_data_sr as source:
        audio = r.record(source)
    try:
        transcription = r.recognize_google(audio, show_all=True, language="cs-CZ")['alternative'][0]['transcript']
    except:
        transcription = 'UNK'

    trans_norm = unicodedata.normalize('NFD', transcription)
    trans_conv = u"".join([c for c in trans_norm if not unicodedata.combining(c)])

    # google translate returns numbers - use written form instead
    numbers_dict = {'4': words[11], '44': words[12], '9': words[15],
                    '11': words[26], '6': words[78]}
    if trans_conv in list(numbers_dict.keys()):
        transcription = numbers_dict[trans_conv]
        trans_norm = unicodedata.normalize('NFD', transcription)
        trans_conv = u"".join([c for c in trans_norm if not unicodedata.combining(c)])

    # get testing word
    testing_samples_ = request.session['testing_samples']
    testing_image_num_ = request.session['testing_image_num']
    testing_image_id = testing_samples_[testing_image_num_ - 1]

    tested_words_ = request.session['tested_words']
    tested_words_.append(words[testing_image_id])
    request.session['tested_words'] = tested_words_

    word_ = request.session['tested_words'][-1]
    word_norm = unicodedata.normalize('NFD', word_)
    word_conv = u"".join([c for c in word_norm if not unicodedata.combining(c)])

    # compare
    if trans_conv.lower() == word_conv.lower():
        is_correct = True
    else:
        is_correct = False

    # save record
    user_id_ = int(request.session['user_id'])
    user_response = UserResponse.create(user_id_, is_correct)
    user_response.save()

    request.session['mode'] = 4

    test_scores_ = request.session['test_scores']
    test_scores_.append(int(is_correct))
    request.session['test_scores'] = test_scores_

    test_responses_path = os.path.join(userdata_path + 'test_responses_' + str(user_id_) + '.npy')
    np.save(test_responses_path, request.session['test_scores'])

    perm_filename = userdata_path + 'recordings/{}_test{}_{}_{}'.format(user_id_, testing_image_num_, word_, int(is_correct))
    with open('{}.wav'.format(perm_filename), mode='bx') as f:
        f.write(request.body)
    os.remove('{}.wav'.format(tmp_filename))

    return HttpResponse('transcription: {}, {}'.format(transcription, is_correct))



def testResults(request):

    score_sum = 0
    finished_users = User.objects.filter(is_finished=True)
    for u in finished_users:
        finished_correct_responses = UserResponse.objects.filter(user_id=u.user_id).filter(is_correct=True)
        score_sum += len(finished_correct_responses)

    user_id_ = request.session['user_id']

    correct_responses = UserResponse.objects.filter(user_id=user_id_).filter(is_correct=True)
    score = len(correct_responses)

    user = User.objects.get(user_id=user_id_)
    user.score = score
    user.is_finished = True
    user.end_time = datetime.datetime.now()
    user.save()

    if len(finished_users) > 0:
        ave_score = float(score_sum) / len(finished_users)
    else:
        ave_score = score

    ucode_ = request.session['ucode']

    context = {'score': score, 'num_testing_images': num_testing_images, 'ave_score': ave_score, 'ucode': ucode_}

    return render(request, 'teacher/testresults.html', context)





