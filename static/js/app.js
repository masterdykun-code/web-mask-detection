const video = document.getElementById('video');
      const iotFrame = document.getElementById('iotFrame');
      const videoBox = document.getElementById('videoBox');
      const videoStage = document.getElementById('videoStage');
      const overlay = document.getElementById('overlay');
      const placeholder = document.getElementById('placeholder');
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      const switchBtn = document.getElementById('switchBtn');
      const cameraSelect = document.getElementById('cameraSelect');
      const sourceSelect = document.getElementById('sourceSelect');
      const iotAccess = document.getElementById('iotAccess');
      const iotViewTokenInput = document.getElementById('iotViewToken');
      const saveIotTokenBtn = document.getElementById('saveIotToken');
      const state = document.getElementById('state');
      const liveStatus = document.getElementById('liveStatus');
      const action = document.getElementById('action');
      const FRAME_GAP_MS = 350;
      const IOT_POLL_MS = 400;
      const MAX_FRAME_WIDTH = 640;
      const METRIC_WINDOW_SIZE = 20;
      const metricSamples = { infer: [], rtt: [], e2e: [] };
      let stream = null;
      let running = false;
      let starting = false;
      let busy = false;
      let timer = null;
      let runToken = 0;
      let detectionPausedUntil = 0;
      let sourceMode = 'laptop';
      let iotSequence = 0;
      let iotMetricSequence = 0;
      let iotInstanceId = '';
      let iotLastFrameAt = 0;

      function setNotice(message, mode) {
        if (!message) {
          action.hidden = true;
          return;
        }
        action.hidden = false;
        action.className = 'notice' + (mode ? ' is-' + mode : '');
        action.querySelector('p').textContent = message;
      }

      function setServerState(message, mode, detail) {
        state.className = 'status' + (mode ? ' is-' + mode : '');
        state.lastElementChild.textContent = message;
        state.title = detail || message;
      }

      function updateMetric(name, value) {
        if (!Number.isFinite(value) || value < 0) return;
        const samples = metricSamples[name];
        samples.push(value);
        if (samples.length > METRIC_WINDOW_SIZE) samples.shift();
        const average = samples.reduce(function (sum, sample) { return sum + sample; }, 0) / samples.length;
        document.getElementById(name).textContent = Math.round(value);
        document.getElementById(name + 'Avg').textContent =
          'TB ' + samples.length + ' frame: ' + Math.round(average) + ' ms';
      }

      function resetMetrics() {
        Object.keys(metricSamples).forEach(function (name) {
          metricSamples[name].length = 0;
          document.getElementById(name).textContent = '—';
          document.getElementById(name + 'Avg').textContent = 'TB —';
        });
        iotMetricSequence = 0;
      }

      function updateModeLabels() {
        const isIot = sourceSelect.value === 'iot';
        document.getElementById('rttLabel').textContent = isIot ? 'ESP32 ↔ Cloud' : 'Local ↔ Cloud';
        document.getElementById('e2eLabel').textContent = isIot ? 'Chụp ESP32 → bbox' : 'Chụp frame → bbox';
        document.getElementById('rttLabel').parentElement.title = isIot
          ? 'Thời gian ESP32 gửi JPEG lên Render và nhận phản hồi HTTP'
          : 'Từ lúc gửi ảnh đến khi nhận và phân tích xong phản hồi JSON';
        document.getElementById('e2eLabel').parentElement.title = isIot
          ? 'Từ thời điểm ESP32 chụp ảnh (đồng bộ NTP) đến khi trình duyệt vẽ bbox'
          : 'Từ lúc chụp frame trên trình duyệt đến khi vẽ bbox';
        cameraSelect.hidden = isIot;
        switchBtn.hidden = isIot;
        iotAccess.hidden = !isIot;
        startBtn.querySelector('.btn-icon').textContent = '▶';
        startBtn.lastChild.textContent = isIot ? 'Kết nối IoT' : 'Bật camera';
      }

      async function refreshHealth() {
        try {
          const response = await fetch('/api/health', { cache: 'no-store' });
          const data = await response.json();
          if (!data.model_loaded) {
            setServerState('Model lỗi', 'error', data.model_error || 'Model chưa sẵn sàng');
            if (!running) setNotice('Model chưa sẵn sàng', 'error');
            return;
          }
          setServerState('Sẵn sàng', 'ready', 'Model đã sẵn sàng');
        } catch (error) {
          setServerState('Mất kết nối', 'error', error.message);
          if (!running) setNotice('Không kết nối được máy chủ', 'error');
        }
      }

      async function listCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cameras = devices.filter(function (device) { return device.kind === 'videoinput'; });
          const selected = cameraSelect.value;
          cameraSelect.replaceChildren(new Option('Camera mặc định', ''));
          cameras.forEach(function (camera, index) {
            const option = new Option(camera.label || ('Camera ' + (index + 1)), camera.deviceId);
            cameraSelect.add(option);
          });
          cameraSelect.value = selected;
        } catch (error) {}
      }

      function resizeOverlay(sourceWidth, sourceHeight) {
        const widthIn = sourceWidth || video.videoWidth || iotFrame.naturalWidth;
        const heightIn = sourceHeight || video.videoHeight || iotFrame.naturalHeight;
        if (!widthIn || !heightIn) return;
        videoBox.style.aspectRatio = widthIn + ' / ' + heightIn;
        const scale = Math.min(
          videoBox.clientWidth / widthIn,
          videoBox.clientHeight / heightIn,
          (window.innerHeight * 0.68) / heightIn
        );
        const width = Math.max(1, Math.round(widthIn * scale));
        const height = Math.max(1, Math.round(heightIn * scale));
        videoStage.style.width = width + 'px';
        videoStage.style.height = height + 'px';
        videoStage.style.display = 'block';
        if (overlay.width !== widthIn) overlay.width = widthIn;
        if (overlay.height !== heightIn) overlay.height = heightIn;
      }

      function normalizedName(value) {
        return String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
      }

      function drawDetections(detections, sourceWidth, sourceHeight) {
        resizeOverlay(sourceWidth, sourceHeight);
        const context = overlay.getContext('2d');
        context.clearRect(0, 0, overlay.width, overlay.height);
        const scaleX = overlay.width / sourceWidth;
        const scaleY = overlay.height / sourceHeight;
        context.lineWidth = Math.max(2, overlay.width / 480);
        context.font = '600 ' + Math.max(14, overlay.width / 82) + 'px system-ui, sans-serif';
        detections.forEach(function (detection) {
          if (!Array.isArray(detection.bbox) || detection.bbox.length !== 4) return;
          const coords = detection.bbox.map(function (value, index) {
            return value * (index % 2 ? scaleY : scaleX);
          });
          const x1 = coords[0], y1 = coords[1], x2 = coords[2], y2 = coords[3];
          if (x2 <= x1 || y2 <= y1) return;
          const alertClass = normalizedName(detection.class_name).includes('khong') ||
            normalizedName(detection.class_name).includes('sai');
          const color = alertClass ? '#ff718b' : '#54e0b3';
          context.strokeStyle = color;
          context.fillStyle = color;
          context.strokeRect(x1, y1, x2 - x1, y2 - y1);
          const label = detection.class_name + ' ' + (detection.confidence * 100).toFixed(1) + '%';
          const textWidth = context.measureText(label).width;
          const labelTop = Math.max(22, y1);
          context.fillRect(x1, labelTop - 22, textWidth + 10, 22);
          context.fillStyle = '#07111f';
          context.fillText(label, x1 + 5, labelTop - 6);
        });
      }

      function scheduleCapture(delay) {
        if (timer) clearTimeout(timer);
        if (!running) return;
        timer = setTimeout(function () {
          timer = null;
          if (sourceMode === 'iot') pollIotLatest();
          else captureAndDetect();
        }, delay === undefined ? FRAME_GAP_MS : delay);
      }

      async function captureAndDetect() {
        if (!running) return;
        if (busy) {
          scheduleCapture(120);
          return;
        }
        const wait = detectionPausedUntil - Date.now();
        if (wait > 0) {
          scheduleCapture(Math.min(wait, 15000));
          return;
        }
        const frameStartedAt = performance.now();
        busy = true;
        const token = runToken;
        try {
          const canvas = document.createElement('canvas');
          const scale = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
          canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
          canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
          canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise(function (resolve) {
            canvas.toBlob(resolve, 'image/jpeg', 0.72);
          });
          if (!blob) throw new Error('Không tạo được ảnh webcam');
          const form = new FormData();
          form.append('frame', blob, 'webcam.jpg');
          const requestStartedAt = performance.now();
          const response = await fetch('/api/detect', { method: 'POST', body: form, cache: 'no-store' });
          const body = await response.text();
          let data;
          try {
            data = body ? JSON.parse(body) : null;
          } catch (error) {
            throw new Error('Phản hồi máy chủ không hợp lệ');
          }
          const rtt = performance.now() - requestStartedAt;
          if (!data) throw new Error('Máy chủ không trả dữ liệu');
          if (token !== runToken) return;
          if (response.status === 429) {
            detectionPausedUntil = Date.now() + 15000;
            setNotice('Tạm nghỉ 15 giây để giảm tải máy chủ', 'error');
            return;
          }
          if (response.status === 503) {
            detectionPausedUntil = Date.now() + 10000;
            setNotice(data.error || 'Model chưa sẵn sàng', 'error');
            return;
          }
          if (!response.ok) throw new Error(data.error || 'Lỗi máy chủ');
          const detections = data.detections || [];
          drawDetections(detections, canvas.width, canvas.height);
          if (data.inference_ms !== undefined) updateMetric('infer', Number(data.inference_ms));
          updateMetric('rtt', rtt);
          updateMetric('e2e', performance.now() - frameStartedAt);
          document.getElementById('count').textContent = detections.length;
          document.getElementById('resultSummary').textContent = detections.length ? 'Đang nhận diện' : 'Không phát hiện';
          renderDetections(detections);
          if (data.message) setNotice(data.message, data.alert ? 'alert' : '');
          else if (data.alert && data.alert_status && data.alert_status.message) setNotice(data.alert_status.message, 'alert');
          else setNotice('', '');
          liveStatus.textContent = 'Đang nhận diện';
          liveStatus.classList.add('is-live');
        } catch (error) {
          if (token === runToken && error.name !== 'AbortError') {
            detectionPausedUntil = Date.now() + 2000;
            setNotice(error.message, 'error');
          }
        } finally {
          busy = false;
          if (running) scheduleCapture(FRAME_GAP_MS);
        }
      }

      async function pollIotLatest() {
        if (!running || sourceMode !== 'iot') return;
        if (busy) {
          scheduleCapture(120);
          return;
        }
        busy = true;
        const token = runToken;
        const pollStartedAt = performance.now();
        let nextDelay = IOT_POLL_MS;
        try {
          const viewToken = sessionStorage.getItem('iotViewToken') || '';
          const query = '?after=' + iotSequence + '&instance=' + encodeURIComponent(iotInstanceId);
          const response = await fetch('/api/iot/latest' + query, {
            headers: { 'X-IoT-View-Token': viewToken },
            cache: 'no-store'
          });
          const body = await response.text();
          let data;
          try {
            data = body ? JSON.parse(body) : null;
          } catch (error) {
            throw new Error('Phản hồi camera IoT không hợp lệ');
          }
          if (token !== runToken) return;
          if (response.status === 401) {
            stopCamera(true);
            setNotice('Mã xem IoT không đúng. Nhập lại mã trên Render.', 'error');
            return;
          }
          if (!response.ok) throw new Error((data && data.error) || 'Không lấy được frame IoT');
          if (data && data.updated && iotInstanceId !== (data.instance_id || '')) iotMetricSequence = 0;
          if (data && data.device_rtt_ms !== null && data.device_rtt_ms !== undefined &&
              data.metrics_sequence > iotMetricSequence) {
            iotMetricSequence = data.metrics_sequence;
            updateMetric('rtt', Number(data.device_rtt_ms));
          }
          if (!data || !data.updated) {
            if (!iotLastFrameAt || performance.now() - iotLastFrameAt > 8000) {
              liveStatus.textContent = 'Đang chờ ESP32-CAM';
              liveStatus.classList.remove('is-live');
              if (iotLastFrameAt) setNotice('Không nhận được frame mới từ ESP32-CAM.', 'error');
            }
            return;
          }

          iotSequence = data.sequence;
          iotInstanceId = data.instance_id || '';
          iotLastFrameAt = performance.now();
          iotFrame.src = 'data:image/jpeg;base64,' + data.frame_jpeg;
          await iotFrame.decode();
          if (token !== runToken) return;
          video.style.display = 'none';
          iotFrame.hidden = false;
          placeholder.style.display = 'none';
          resizeOverlay(data.width, data.height);
          const detections = data.detections || [];
          drawDetections(detections, data.width, data.height);
          if (data.inference_ms !== undefined) updateMetric('infer', Number(data.inference_ms));
          if (data.captured_at_ms) {
            document.getElementById('e2eLabel').textContent = 'Chụp ESP32 → bbox';
            document.getElementById('e2eLabel').parentElement.title =
              'Từ thời điểm ESP32 chụp ảnh (đồng bộ NTP) đến khi trình duyệt vẽ bbox';
            updateMetric('e2e', Math.max(0, Date.now() - Number(data.captured_at_ms)));
          } else {
            document.getElementById('e2eLabel').textContent = 'Cloud → bbox';
            document.getElementById('e2eLabel').parentElement.title =
              'Từ lúc trình duyệt nhận frame từ Render đến khi vẽ bbox; ESP32 chưa đồng bộ được NTP';
            updateMetric('e2e', performance.now() - pollStartedAt);
          }
          document.getElementById('count').textContent = detections.length;
          document.getElementById('resultSummary').textContent = detections.length ? 'Đang nhận diện' : 'Không phát hiện';
          renderDetections(detections);
          if (data.message) setNotice(data.message, data.alert ? 'alert' : '');
          else if (data.alert && data.alert_status && data.alert_status.message) setNotice(data.alert_status.message, 'alert');
          else setNotice('', '');
          liveStatus.textContent = 'Đang nhận diện IoT';
          liveStatus.classList.add('is-live');
          setServerState('Sẵn sàng', 'ready', 'ESP32-CAM đang gửi ảnh trực tiếp lên Render');
        } catch (error) {
          if (token === runToken) {
            nextDelay = 2000;
            setNotice(error.message, 'error');
          }
        } finally {
          busy = false;
          if (running && sourceMode === 'iot') scheduleCapture(nextDelay);
        }
      }

      async function startIotMode() {
        if (running || starting) return;
        const viewToken = iotViewTokenInput.value.trim() || sessionStorage.getItem('iotViewToken') || '';
        if (!viewToken) {
          setNotice('Nhập mã xem IoT đã cấu hình trên Render.', 'error');
          return;
        }
        sessionStorage.setItem('iotViewToken', viewToken);
        iotViewTokenInput.value = viewToken;
        starting = true;
        startBtn.disabled = true;
        setNotice('Đang kết nối camera IoT…', '');
        try {
          sourceMode = 'iot';
          resetMetrics();
          iotSequence = 0;
          iotInstanceId = '';
          iotLastFrameAt = 0;
          running = true;
          runToken += 1;
          video.srcObject = null;
          video.style.display = 'none';
          iotFrame.hidden = true;
          videoStage.style.display = 'none';
          placeholder.style.display = 'block';
          placeholder.textContent = 'Đang chờ frame từ ESP32-CAM…';
          startBtn.disabled = true;
          stopBtn.disabled = false;
          liveStatus.textContent = 'Đang chờ ESP32-CAM';
          liveStatus.classList.add('is-live');
          scheduleCapture(0);
        } finally {
          starting = false;
        }
      }

      function startSelectedSource() {
        if (sourceSelect.value === 'iot') startIotMode();
        else startCamera();
      }

      function renderDetections(detections) {
        const list = document.getElementById('detections');
        list.replaceChildren();
        if (!detections.length) {
          const empty = document.createElement('div');
          empty.className = 'empty';
          empty.textContent = 'Chưa phát hiện đối tượng';
          list.appendChild(empty);
          return;
        }
        detections.forEach(function (detection) {
          const alertClass = normalizedName(detection.class_name).includes('khong') ||
            normalizedName(detection.class_name).includes('sai');
          const row = document.createElement('div');
          row.className = 'detection-row' + (alertClass ? ' is-alert' : '');
          const name = document.createElement('span');
          name.className = 'detection-name';
          const dot = document.createElement('i');
          dot.className = 'detection-dot';
          const label = document.createElement('span');
          label.textContent = detection.class_name;
          name.append(dot, label);
          const confidence = document.createElement('span');
          confidence.className = 'confidence';
          confidence.textContent = (detection.confidence * 100).toFixed(1) + '%';
          row.append(name, confidence);
          list.appendChild(row);
        });
      }

      async function startCamera() {
        if (running || starting) return;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          setNotice('Trình duyệt không hỗ trợ camera', 'error');
          return;
        }
        starting = true;
        sourceMode = 'laptop';
        resetMetrics();
        startBtn.disabled = true;
        setNotice('Đang mở camera…', '');
        try {
          const constraints = {
            audio: false,
            video: cameraSelect.value
              ? { deviceId: { exact: cameraSelect.value }, width: { ideal: 1280 }, height: { ideal: 720 } }
              : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
          };
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          video.srcObject = stream;
          iotFrame.hidden = true;
          video.style.display = 'block';
          placeholder.style.display = 'none';
          await new Promise(function (resolve) {
            if (video.readyState >= 1) resolve();
            else video.addEventListener('loadedmetadata', resolve, { once: true });
          });
          await video.play();
          if (!video.videoWidth || !video.videoHeight) throw new Error('Không lấy được hình ảnh từ camera');
          resizeOverlay();
          running = true;
          runToken += 1;
          startBtn.disabled = true;
          stopBtn.disabled = false;
          liveStatus.textContent = 'Đang nhận diện';
          liveStatus.classList.add('is-live');
          setNotice('Đang nhận diện', '');
          await listCameras();
          scheduleCapture(0);
        } catch (error) {
          stopCamera(true);
          setNotice('Không mở được camera: ' + error.message, 'error');
        } finally {
          starting = false;
          if (!running) startBtn.disabled = false;
        }
      }

      function stopCamera(showMessage) {
        running = false;
        runToken += 1;
        if (timer) clearTimeout(timer);
        timer = null;
        if (stream) {
          stream.getTracks().forEach(function (track) { track.stop(); });
          stream = null;
        }
        video.srcObject = null;
        video.style.display = 'none';
        iotFrame.hidden = true;
        iotFrame.removeAttribute('src');
        videoStage.style.display = 'none';
        overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
        placeholder.style.display = 'block';
        placeholder.textContent = showMessage ? 'Camera đã tắt' : 'Đang chuyển camera…';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        liveStatus.textContent = 'Đang tắt';
        liveStatus.classList.remove('is-live');
        setNotice('', '');
      }

      async function switchCamera() {
        await listCameras();
        if (cameraSelect.options.length <= 1) return;
        cameraSelect.selectedIndex = (cameraSelect.selectedIndex + 1) % cameraSelect.options.length;
        if (running) {
          stopCamera(false);
          await startCamera();
        }
      }

      video.addEventListener('loadedmetadata', resizeOverlay);
      window.addEventListener('resize', resizeOverlay);
      startBtn.addEventListener('click', startSelectedSource);
      stopBtn.addEventListener('click', function () { stopCamera(true); });
      switchBtn.addEventListener('click', switchCamera);
      sourceSelect.addEventListener('change', function () {
        if (running) stopCamera(false);
        updateModeLabels();
        resetMetrics();
        placeholder.textContent = sourceSelect.value === 'iot'
          ? 'Bật chế độ IoT để chờ ESP32-CAM'
          : 'Bật camera để bắt đầu';
      });
      saveIotTokenBtn.addEventListener('click', function () {
        const value = iotViewTokenInput.value.trim();
        if (!value) {
          setNotice('Nhập mã xem IoT trước.', 'error');
          return;
        }
        sessionStorage.setItem('iotViewToken', value);
        setNotice('Đã lưu mã xem trong tab này.', '');
      });
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', listCameras);
      }
      iotViewTokenInput.value = sessionStorage.getItem('iotViewToken') || '';
      updateModeLabels();
      refreshHealth();
      listCameras();
