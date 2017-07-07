package vgg.vgg163;

import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void run(View view){
        AssetManager assetManager = getAssets();
        TextView tv = (TextView)findViewById(R.id.textView);
        tv.setText(new tensorflow().runalexnet(assetManager));
    }

    public void run2(View view){
        AssetManager assetManager = getAssets();
        TextView tv = (TextView)findViewById(R.id.textView);
        tv.setText("!!!");
    }
}
