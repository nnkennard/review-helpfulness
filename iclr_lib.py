import json
import openreview
import os
import tqdm

INVITATION_MAP = {
    f"iclr_{year}": f"ICLR.cc/{year}/Conference/-/Blind_Submission"
    for year in range(2018, 2022)
}

GUEST_CLIENT = openreview.Client(baseurl="https://api.openreview.net")


def get_replies(note, discussion_notes):
    return [x for x in discussion_notes if x.replyto == note.id]


def get_longest_thread(top_note, discussion_notes):
    threads = [[top_note]]
    while True:
        new_threads = []
        for thread in threads:
            thread_last = thread[-1]
            candidates = [
                c for c in get_replies(thread_last, discussion_notes)
                if c.signatures == thread_last.signatures
            ]
            for candidate in candidates:
                new_threads.append(thread + [candidate])
        if not new_threads:
            break
        threads = new_threads
    return max(threads, key=lambda x: len(x))


def get_review_threads(discussion_notes):
    reviews = []
    for note in discussion_notes:
        if note.replyto == note.forum and "review" in note.content:
            reviews.append((note.signatures[0],
                            get_longest_thread(note, discussion_notes)))
    return [x[1] for x in sorted(reviews, key=lambda x: (x[0], x[1][0].id))]


def get_review_data(discussion_notes, metadata):
    review_threads = get_review_threads(discussion_notes)
    review_data = []
    for i, thread in enumerate(review_threads):

        review_data.append({
            "identifier":
            f"{thread[0].forum}___{i}",
            "reviewer":
            thread[0].signatures[0].split("/")[-1],
            "rating":
            int(thread[0].content["rating"].split(":")[0]),
            "text":
            "\n\n".join([
                n.content.get("review", n.content.get("comment"))
                for n in thread
            ]),
            "metadata":
            metadata,
        })
    return review_data


def get_decision(discussion_notes):
    for note in discussion_notes:
        maybe_decision = note.content.get(
            "decision", note.content.get("recommendation", None))
        if maybe_decision is not None:
            return maybe_decision
    assert False


def get_iclr_data(output_dir):
    for conference, invitation in INVITATION_MAP.items():
        os.makedirs(output_dir, exist_ok=True)
        forum_notes = list(
            openreview.tools.iterget_notes(GUEST_CLIENT,
                                           invitation=invitation))
        for forum_note in tqdm.tqdm(forum_notes):
            discussion_notes = GUEST_CLIENT.get_notes(forum=forum_note.id)
            metadata = {
                "authors": [],
                "decision": get_decision(discussion_notes),
                "conference": conference,
            }
            for note in discussion_notes:
                if note.id == note.forum:
                    metadata["forum_id"] = note.id
                    metadata["title"] = note.content["title"]
                    for author_name, author_id in zip(
                            note.content["authors"],
                            note.content["authorids"]):
                        metadata["authors"].append({
                            "name": author_name,
                            "author_id": author_id
                        })
            assert "forum_id" in metadata
            for review in get_review_data(discussion_notes, metadata):
                with open(f'{output_dir}/{review["identifier"]}.json',
                          "w") as f:
                    json.dump(review, f)
