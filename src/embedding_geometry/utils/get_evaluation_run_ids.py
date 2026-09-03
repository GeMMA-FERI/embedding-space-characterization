from semora.storage import Database


def main() -> None:
    db = Database("data/newspapers.sqlite")

    try:
        db.initialize()

        for embedding_run_id in db.get_non_noop_embedding_run_ids():
            print(embedding_run_id)

    finally:
        db.close()


if __name__ == "__main__":
    main()
