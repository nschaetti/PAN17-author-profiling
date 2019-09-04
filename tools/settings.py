#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Global settings
n_authors = 15
n_pretrain_authors = 50 - n_authors
k = 10

# Authors
authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw', u'MichaelConnor',
           u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison', u'HeatherScoffield', u'BradDorfman',
           u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning', u'ScottHillis', u'FumikoFujisaki', u'TimFarrand',
           u'SarahDavison', u'AaronPressman', u'JohnMastrini', u'NickLouth', u'PierreTran', u'AlexanderSmith',
           u'MatthewBunce', u'KouroshKarimkhany', u'JimGilchrist', u'DarrenSchuettler', u'TanEeLyn', u'JoeOrtiz',
           u'MureDickie', u'EdnaFernandes', u'JoWinterbottom', u'RogerFillion', u'BenjaminKangLim', u"LynneO'Donnell",
           u'JonathanBirt', u'BernardHickey', u'RobinSidel', u'AlanCrosby', u'LydiaZajc', u'PeterHumphrey',
           u'KeithWeir', u'EricAuchard', u'TheresePoletti', u'KarlPenhaul', u'SimonCowell', u'JaneMacartney',
           u'SamuelPerry', u'MarkBendeich']
train_authors = [u'JanLopatka', u'WilliamKazer', u'MarcelMichelson', u'KirstinRidley', u'GrahamEarnshaw',
                 u'MichaelConnor', u'MartinWolk', u'ToddNissen', u'PatriciaCommins', u'KevinMorrison',
                 u'HeatherScoffield', u'BradDorfman', u'DavidLawder', u'KevinDrawbaugh', u'LynnleyBrowning']
pretrain_authors = [a for a in authors if a not in train_authors]

# Glove settings
glove_embedding_dim = 300

# Length
pos_length = 1356
fw_length = 477
wv_length = 1356
c1_length = 6901
c2_length = 6900
c3_length = 6899

# Voc size
voc_size = dict()
voc_size['pos'] = 17
voc_size['fw'] = 193
voc_size['wv'] = 27396
voc_size['c1'] = 85
voc_size['c2'] = 2764
voc_size['c3'] = 22097
voc_size['ce'] = 85

# Input dim
input_dims = dict()
input_dims['wv'] = 300
input_dims['fw'] = 300
input_dims['c1'] = 10
input_dims['c2'] = 30
input_dims['c3'] = 60

# Character Encoder
ce_text_length = 15
