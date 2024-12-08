# Intermediate Hand-In for Similarity Modeling 1 & 2

For the intermediate hand-in, we are submitting our data loader scripts, which can be found in the `/scripts` directory.

## Submitted Scripts

1. **`setup.py`**
   - Loads the annotations from the ground truth for all three videos.
   - Extracts the frames and audio for each video.
   - Aligns the extracted frames and audio with the loaded annotations.

2. **`load_data.py`**
   - Checks whether frames and audio are extracted, if not -> runs `setup.py`
   - Loads the extracted audio and frames into memory
   - Frames and audio are loaded as dictionaries
        - key: video index 
        - value: tuples of frame/segment index and the file path.

3. **`trim_video.py`**  
   - Trims the videos to the first `duration` seconds.
   - For experimentation, videos are trimmed to 1 minute to reduce testing time.
   - The full pipeline (code and models) will be executed on the complete videos.

Finally, we test the dataloader in the setup_test.ipynb file.

## Time sheet for the intermediate hand-in

**Kravchenko Oleksandra**
<table>
<thead>
  <tr>
    <th>Date</th>
    <th>Task</th>
    <th>Hours</th>

  </tr>
</thead>
<tbody>
  <tr>
    <td>24.10.2024</td>
    <td>Initial Team Meeting</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td>01.12.2024</td>
    <td>Second Team Meeting - Discussion of extraction approach and work split</td>
    <td>0.5</td>
  </tr>
      <tr>
    <td>08.12.2024</td>
    <td>Third Team Meeting</td>
    <td>0.5</td>
  </tr>
</tbody>
</table>

---

**SAKKA Mahmoud Abdussalem**
<table>
<thead>
  <tr>
    <th>Date</th>
    <th>Task</th>
    <th>Hours</th>

  </tr>
</thead>
<tbody>
  <tr>
    <td>24.10.2024</td>
    <td>Initial Team Meeting</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td>01.12.2024</td>
    <td>Second Team Meeting - Discussion of extraction approach and work split</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td>06.12.2024</td>
    <td>Initial exploration of material, implementation of trim_video.py and audio_extraction</td>
    <td>4</td>
  </tr>
  <tr>
    <td>07.12.2024</td>
    <td>Implementation of load_data.py</td>
    <td>2</td>
  </tr>
    <tr>
    <td>08.12.2024</td>
    <td>Third Team Meeting</td>
    <td>0.5</td>
  </tr>
</tbody>
</table>


# Setup outside of find_kermit
- ground_truth_data (repository with muppets avi files)
    - audio (repository for audio)
    - frames (repository for extracted frames before sorting)
    - train_data 
    - test_data 
    - validation_data

