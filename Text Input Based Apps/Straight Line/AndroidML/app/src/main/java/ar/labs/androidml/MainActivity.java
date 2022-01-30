
package ar.labs.androidml;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import ar.labs.androidml.ml.Model1a;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btn= findViewById(R.id.button);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try{
                    EditText inputEditText;
                    inputEditText = findViewById(R.id.editTextNumberDecimal);
                    Float data= Float.parseFloat(inputEditText.getText().toString());
                    ByteBuffer byteBuffer= ByteBuffer.allocateDirect(1*4);
                    byteBuffer.order(ByteOrder.nativeOrder()); // order in little endian format
                    byteBuffer.putFloat(data);
                    byteBuffer.rewind();



                    Model1a model = Model1a.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 1}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model1a.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    TextView tv= findViewById(R.id.textView);
                    float[] data1=outputFeature0.getFloatArray();
                    tv.setText(String.valueOf(data1[0]));


                    model.close();

                }
                catch (Exception e)
                {
                    Toast.makeText(getApplicationContext(),"Issue...",Toast.LENGTH_LONG).show();
                }
            }
        });
    }
}