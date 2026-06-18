# Running the MLB Forecaster on Google Cloud (beginner guide)

This guide assumes **zero prior Google Cloud experience**. You will rent one small
Linux machine ("VM"), install the app, run it, and optionally schedule it to update
itself daily. Every command is copy-paste ready. Placeholders look like
`YOUR_PROJECT_ID` — replace them.

**What it costs:** new Google Cloud accounts get **$300 free credit**. The machine
in this guide (`e2-medium`) costs about **$0.03/hour (~$24/month if left on 24/7)**.
If you **stop** the VM when you're done, you pay only for disk (~$1/month). Part 5
shows how to stop/delete it so you don't get surprise charges.

---

## Part 0 — One-time account setup (Console, in your browser)

1. Go to **https://console.cloud.google.com** and sign in with your Google account.
2. **Create a project:** top bar → project dropdown → **New Project** → name it
   `mlb538` → **Create**. Wait ~30 seconds, then select it in the dropdown.
3. **Enable billing:** left menu (☰) → **Billing** → **Link a billing account**
   (set one up if prompted; you get the $300 free credit). The project must have
   billing enabled or nothing will run.

That's all the clicking. Everything else is copy-paste in **Cloud Shell**.

---

## Part 1 — Open Cloud Shell and enable the services

**Cloud Shell** is a free Linux terminal in your browser with all Google tools
pre-installed and already logged in. No installs on your own computer.

1. In the Console, click the **terminal icon `>_`** in the top-right. A terminal
   opens at the bottom. If asked to authorize, click **Authorize**.
2. Point it at your project and turn on the services we need (Compute Engine =
   virtual machines, Cloud Storage = file storage):

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com storage.googleapis.com
```

(The `enable` step can take a minute the first time.)

---

## Part 2 — Create the machine and install the app

### 2a. Create the VM (run in Cloud Shell)

Paste this as **one single line** (multi-line commands with `\` can get mangled when
pasted into the browser terminal, which causes an `--image-project` / `--image-family`
error):

```bash
gcloud compute instances create mlb538 --zone=us-central1-a --machine-type=e2-medium --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=30GB
```

This makes a small Linux computer named `mlb538` with a 30 GB disk (enough for the
full ~4 GB raw data cache plus room to spare).

### 2b. Log into the machine

```bash
gcloud compute ssh mlb538 --zone=us-central1-a
```

The first time, it creates SSH keys for you — just press Enter through any prompts
(leave the passphrase empty). Your prompt will change to `...@mlb538:~$`, meaning
you are now **inside the cloud machine**. All the following commands run there.

### 2c. Install Python, Git, and the app

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git
git clone https://github.com/storeyboxdev/MLB538.git
cd MLB538
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2d. Run the pipeline

Start with a **fast schedule-only run** to confirm everything works (seconds, ~20 MB):

```bash
mlbfc scrape --no-boxscores      # pull schedules + results for the default seasons
mlbfc rate                       # build Elo ratings
mlbfc train                      # train the model
mlbfc forecast --season 2025 --sims 10000
```

You should see a table of playoff/World Series odds. 🎉

For the **full model with pitcher data** (slower — about 3 min/season, ~4 GB total):

```bash
mlbfc scrape                     # now pulls starter box scores too
mlbfc train --fit-elo            # fit Elo knobs + train classifier + margin model
mlbfc backtest                   # honest accuracy check
mlbfc forecast --season 2025 --sims 10000
```

To forecast the **live, in-progress season**, scrape it first:

```bash
mlbfc scrape --seasons 2026
mlbfc forecast --season 2026 --sims 10000
```

Outputs land in `data/output/` (`forecast.json`, `ratings.csv`, `backtest.json`).

---

## Part 3 — Get the results off the machine

**Quick look (on the VM):**
```bash
cat data/output/forecast.json
```

**Download to Cloud Shell / your computer:** open a *new* Cloud Shell tab (don't
SSH) and run:
```bash
gcloud compute scp mlb538:~/MLB538/data/output/forecast.json . --zone=us-central1-a
```

**Save to a Cloud Storage bucket** (best for sharing / feeding other apps). Create a
bucket once (name must be globally unique — use your project id):
```bash
gcloud storage buckets create gs://YOUR_PROJECT_ID-mlb538 --location=us-central1
```
Then, from the VM, copy outputs up any time:
```bash
gcloud storage cp data/output/*.json gs://YOUR_PROJECT_ID-mlb538/
```

---

## Part 4 — (Optional) Update itself every day

The app has an `mlbfc update` command that pulls the latest games, advances the
ratings, and retrains on a cadence. The simplest way to run it daily is **cron** on
the VM (the VM must be left running for this).

On the VM:
```bash
# Create a tiny script that runs the daily update + a fresh forecast + uploads it
cat > ~/daily.sh <<'EOF'
#!/usr/bin/env bash
cd ~/MLB538
source .venv/bin/activate
mlbfc update --season 2026
mlbfc forecast --season 2026 --sims 10000
gcloud storage cp data/output/forecast.json gs://YOUR_PROJECT_ID-mlb538/
EOF
chmod +x ~/daily.sh

# Schedule it for 6:00 AM every day
( crontab -l 2>/dev/null; echo "0 6 * * * /home/$USER/daily.sh >> /home/$USER/daily.log 2>&1" ) | crontab -
```

Check it later with `cat ~/daily.log`.

> A more "cloud-native" alternative (no always-on VM) is a **Cloud Run Job** on a
> **Cloud Scheduler** trigger. It's cheaper and tidier but requires packaging the app
> as a Docker container. Ask and I'll add a `Dockerfile` + the exact commands.

---

## Part 5 — Control costs (do this!)

The VM bills while it's **running**. When you're done for the day:

```bash
# From Cloud Shell (not from inside the VM):
gcloud compute instances stop mlb538 --zone=us-central1-a     # stops billing for compute
gcloud compute instances start mlb538 --zone=us-central1-a    # turn it back on later
```

When you're completely finished and want to stop all charges:
```bash
gcloud compute instances delete mlb538 --zone=us-central1-a
```

Set a **budget alert** so you're emailed if spend approaches a limit:
Console → **Billing** → **Budgets & alerts** → **Create budget** → set e.g. $20.

---

## Part 6 — Land scraped data in a bucket, then ingest into BigQuery

The scraper can upload each season's processed data as **Parquet** to a Cloud
Storage bucket as it goes (typed, ~90 KB/season). BigQuery then ingests from the
bucket. The 4 GB raw cache never leaves the VM.

### 6a. Create a bucket and enable BigQuery (Cloud Shell)

```bash
gcloud storage buckets create gs://YOUR_PROJECT_ID-mlb538 --location=us-central1
gcloud services enable bigquery.googleapis.com
```

### 6b. Give the VM permission to write to Cloud Storage (Cloud Shell)

VMs are created read-only for storage by default, so widen the scope once:

```bash
gcloud compute instances stop mlb538 --zone=us-central1-a
gcloud compute instances set-service-account mlb538 --zone=us-central1-a --scopes=cloud-platform
gcloud compute instances start mlb538 --zone=us-central1-a
```

Then SSH back in: `gcloud compute ssh mlb538 --zone=us-central1-a`.

### 6c. Turn on the GCS sink and (re)scrape (on the VM)

```bash
cd ~/MLB538 && source .venv/bin/activate
pip install -e ".[gcp]"                          # adds google-cloud-storage + pyarrow
export MLB538_GCS_BUCKET=YOUR_PROJECT_ID-mlb538   # or set data.gcs_bucket in config.yaml
mlbfc scrape                                      # cached games are instant; now uploads Parquet
```

Verify the files landed:
```bash
gcloud storage ls gs://YOUR_PROJECT_ID-mlb538/mlb538/processed/
```

> Tip: add `export MLB538_GCS_BUCKET=...` to `~/.bashrc` so it's always set, and the
> daily cron job (Part 4) will upload automatically.

### 6d. Ingest into BigQuery (Cloud Shell)

```bash
bq --location=us-central1 mk --dataset YOUR_PROJECT_ID:mlb538
bq load --source_format=PARQUET --replace --autodetect \
  YOUR_PROJECT_ID:mlb538.games \
  "gs://YOUR_PROJECT_ID-mlb538/mlb538/processed/games_*.parquet"
```

`--replace` rebuilds the table from all season files each run, so re-running after a
daily scrape keeps BigQuery current. Sanity-check it:

```bash
bq query --use_legacy_sql=false \
  "SELECT season, COUNT(*) AS games FROM mlb538.games GROUP BY season ORDER BY season"
```

> **Zero-maintenance alternative:** an *external table* queries the Parquet in place,
> so new uploads appear automatically with no reload:
> ```bash
> bq mkdef --source_format=PARQUET "gs://YOUR_PROJECT_ID-mlb538/mlb538/processed/games_*.parquet" > games_def.json
> bq mk --external_table_definition=games_def.json YOUR_PROJECT_ID:mlb538.games
> ```

### 6e. Next: ratings & forecast tables, and the LLM

Today only the `games` table is uploaded. The `ratings.csv` and `forecast.json`
outputs can be uploaded the same way (a small addition to `pipeline.rate` /
`pipeline.forecast`) so the LLM can answer "current rating" and "playoff odds"
questions. The LLM app then gets a BigQuery query tool over these tables.

---

## Troubleshooting

- **`gcloud: command not found`** — you're inside the VM, not Cloud Shell. Some
  `gcloud` commands (create/stop/scp) run from Cloud Shell; the `mlbfc` commands run
  on the VM. Open a separate Cloud Shell tab for `gcloud compute ...`.
- **`Permission denied` / billing errors** — the project needs billing enabled
  (Part 0, step 3) and the APIs enabled (Part 1).
- **SSH won't connect** — wait ~30 seconds after creating the VM, then retry
  `gcloud compute ssh mlb538 --zone=us-central1-a`.
- **Out of disk during full scrape** — the raw cache is ~4 GB; the 30 GB disk is
  plenty. If you chose a smaller disk, recreate the VM with `--boot-disk-size=30GB`.
- **`mlbfc: command not found`** — run `source .venv/bin/activate` first (and `cd
  ~/MLB538`).
