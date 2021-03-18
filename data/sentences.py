sentences = [
        # # VCTK_unseen_reference
        # # parallel
        # "having guidelines in advance is helpful.", #3
        # "another high street retailer was not so lucky.", #1
        # "people have been placed in a state of alarm.", #1
        # "finding suitable replacements would not be easy.", #2
        # "how do you take them away?", #21
        # "it particularly increases under-age drinking, he claimed.", #3
        # "millions of british jobs depend on europe.", #5
        # "corporate banking would be based in edinburgh.", #6
        # "throughout the centuries people have explained the rainbow in various ways.", #86
        # "he already had had complaints.", #2
        # "connell was standing there, ordering her to get up.", #2
        # "he wasn't really doing anything, so he wanted out.", #2
        # "as long as it is possible, we will keep going.", #2
        # "both were later released after a check-up.", #1
        # "millions of british jobs depend on europe.", #5
        # "i did not concentrate on my performance.", #3
        # "that period was a struggle.", #7
        # "yesterday, he continued to keep a low profile.", #13
        # "it will now relate to all public bodies in scotland.", #10
        # "the final decision was between scotland and the republic of ireland.", #7
        # "it will take place in july.", #7
        # "blair is very positive at european councils.", #13
        # "the norsemen considered the rainbow as a bridge over which the gods passed from the earth to their homes in the sky.", #1
        # # nonparallel
        # 'when a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.',
        # 'since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows.',
        # 'other members of the family were too upset to comment last night.',
        # 'the guidelines are expected to be finalised before next spring.',
        # 'for the meantime, though, the signs are good.',
        # 'these take the shape of a long round arch, and its path high above, and its two ends apparently beyond the horizon.', #1
        # 'the senior management team will be drawn from both companies.', #1
        # 'but do not rely on it.', #1


        # VCTK_seen_reference
        # parallel
        "i'm taking each day as it comes.",
        "this is a milestone in the modernisation of the scottish prosecution service.",
        "ask her to bring these things with her from the store.",
        "this is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "i had relied on him.",
        "we also need a small plastic snake and a big toy frog for the kids.",
        "it makes no difference to their friendship.",
        "murray financial has fallen at the first hurdle.",
        "drugs are used a lot at the fishing, not just cannabis.",
        "the difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases.",
        "people come into the borders for the beauty of the background.",
        "nothing is lost, everything is recycled.",
        # nonparallel
        # "do i have a favourite?",
        # "is it in the right place?",
        # "does that put pressure on us?",
        # "hat was the matter for concern?",
        # "was everything done to save people?",
        # "what kind of man does that, mr dick?",
        "people look, but no one ever finds it.",
        "others have tried to explain the phenomenon physically.",
        "many complicated ideas about the rainbow have been formed.",
        "some have accepted it as a miracle without physical explanation.",
        "throughout the centuries people have explained the rainbow in various ways.",
        "when the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
        "she can scoop these things into three red bags, and we will go meet her wednesday at the train station.",
        "if the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow.",


        # # VCTK_val_reference
        # "we also need a small plastic snake and a big toy frog for the kids.",
        # "this is a milestone in the modernisation of the scottish prosecution service.",
        # "all businesses continue to trade.",
        # "nothing is lost, everything is recycled.",
        # "one mp said he feared the job losses were only the beginning.",
        # "the difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases.",
        # "the party has never fully recovered.",
        # "the confidence is low, but it is a difficult thing to understand.",
        # "people come into the borders for the beauty of the background. ",
        # "the next eight weeks are critical to us.",
        # "this is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        # "i had relied on him. ",

        # # NonParallelRefs
        # "dogs are sitting by the door!",
        # "matthew cuthbert is surprised",
        # "has never been surpassed.",
        # "in being comparatively modern.",
        # "ambitious hopes, which had seemed to be extinguished, revived in his bosom.",
        # "after a pause bechamel went back to the dining room.",
        # "Now it was finished - that is to say the design - she must stitch it together .",
        # "when we first met here we were younger than our girls are now.",
        # "oh my god, he's lost it. he's totally lost it.",
        # "you must know said margolotte when they were all seated together on the broad window seat that my husband foolishly gave away all the powder of life he first made to old mombi the witch who used to live in the country of the gillikins to the north of here.",
        # "And it is worth mention in passing that, as an example of fine typography,",
        # "Printing, then, for our purpose, may be considered as the art of making books by means of movable types.",
        # "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        # "Advanced text to speech models such as Fast Speech can synthesize speech significantly faster than previous auto regressive models with comparable quality. The training of Fast Speech model relies on an auto regressive teacher model for duration prediction and knowledge distillation, which can ease the one to many mapping problem in T T S. However, Fast Speech has several disadvantages, 1, the teacher student distillation pipeline is complicated, 2, the duration extracted from the teacher model is not accurate enough, and the target mel spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality.",
        # "in the aftermath of this storm, we were thrown back to the east. away went any hope of",
        # "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
        # "produced the block books, which were the immediate predecessors of the true printed book,",
        # "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
        # "the earliest book printed with movable types, the Gutenberg, or \"forty-two line Bible\" of about 1455,",
        # "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,"
    ]