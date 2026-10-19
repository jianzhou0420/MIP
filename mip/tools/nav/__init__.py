"""The discrete navigation tools over an env server (EmbodiedScoreEnv), every line.

The Minimal interface is ``[nav.observe, nav.step]``; the EQA lines add ``nav.answer``. The tools
shape themselves by the placement's task word (Session.task):

    nav      VLN-CE R2R / RxR — instruction in the briefing, STOP terminal
    objnav   ObjectNav HM3D/MP3D — category in the briefing, STOP terminal
    eqa      HM-EQA — question + choices in the briefing, answer(letter)

The action table is a prefix of 0 STOP · 1 FORWARD · 2 LEFT · 3 RIGHT · 4 LOOK_UP ·
5 LOOK_DOWN, its size the line's (Session.n_actions: 4 on the paper protocol, 6 STANDARD);
the EQA lines drop STOP. The agent never sees SR/SPL, reward, pose, depth or panoramas.
"""
