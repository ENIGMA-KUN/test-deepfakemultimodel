$formFile = "upload_form.html"

# Create a simple HTML form
$html = @"
<!DOCTYPE html>
<html>
<head>
    <title>File Upload</title>
</head>
<body>
    <h2>Upload File Test</h2>
    <form action="http://localhost:8000/api/v1/upload/upload" method="POST" enctype="multipart/form-data">
        <div>
            <label for="file">Select File:</label>
            <input type="file" name="file" id="file" required>
        </div>
        <div>
            <label for="media_type">Media Type:</label>
            <select name="media_type" id="media_type">
                <option value="image">Image</option>
                <option value="audio">Audio</option>
                <option value="video">Video</option>
            </select>
        </div>
        <div>
            <label for="detailed_analysis">Detailed Analysis:</label>
            <input type="checkbox" name="detailed_analysis" id="detailed_analysis" value="true" checked>
        </div>
        <div>
            <label for="confidence_threshold">Confidence Threshold:</label>
            <input type="number" name="confidence_threshold" id="confidence_threshold" value="0.5" min="0" max="1" step="0.01">
        </div>
        <div>
            <button type="submit">Upload</button>
        </div>
    </form>
</body>
</html>
"@

# Write to file
Set-Content -Path $formFile -Value $html

# Open the form in the default browser
Write-Host "Opening HTML form in your default browser..."
Start-Process $formFile

Write-Host "Please complete the form in your browser to upload the file." 